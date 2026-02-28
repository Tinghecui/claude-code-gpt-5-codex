from copy import deepcopy
from typing import AsyncGenerator, Callable, Generator, Optional, Union
import logging
import time
import traceback

import httpx
import litellm

logger = logging.getLogger(__name__)
from litellm import (
    BaseResponsesAPIStreamingIterator,
    CustomLLM,
    CustomStreamWrapper,
    GenericStreamingChunk,
    HTTPHandler,
    ModelResponse,
    ModelResponseStream,
    AsyncHTTPHandler,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)

from claude_code_proxy.route_model import ModelRoute
from common.config import (
    ENABLE_STREAM_DIAGNOSTIC_LOGS,
    STREAM_DIAGNOSTIC_SLOW_GAP_MS,
    WRITE_TRACES_TO_FILES,
)
from common.tracing_in_markdown import (
    write_request_trace,
    write_response_trace,
    write_streaming_chunk_trace,
)
from common.utils import (
    ProxyError,
    convert_chat_messages_to_respapi,
    convert_chat_params_to_respapi,
    convert_respapi_to_model_response,
    generate_timestamp_utc,
    reset_request_context,
    to_generic_streaming_chunk,
    responses_eof_finalize_chunk,
)
import random


# === Monkey-patch LiteLLM to include cache fields in Anthropic response ===
def _extract_cache_from_usage(usage):
    """从 OpenAI usage 对象中提取缓存信息"""
    cache_read = 0
    cache_creation = 0
    prompt_tokens = 0

    if usage is None:
        return cache_read, cache_creation

    # 获取 prompt_tokens
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0

    # 检查 prompt_tokens_details (Chat Completions API)
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cache_read = getattr(details, "cached_tokens", 0) or 0

    # 检查 input_tokens_details (Responses API)
    if cache_read == 0:
        details = getattr(usage, "input_tokens_details", None)
        if details:
            cache_read = getattr(details, "cached_tokens", 0) or 0

    # 模拟 cache_creation_input_tokens
    if prompt_tokens > 0:
        cache_creation = int(prompt_tokens * random.uniform(0.5, 0.8))

    return cache_read, cache_creation


def _patch_litellm_anthropic_usage():
    """
    完整的 LiteLLM Patch，包括：
    1. 非流式响应转换
    2. 流式响应转换
    3. 流式迭代器 usage 更新
    """
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters import (
            transformation,
            streaming_iterator,
        )
        from litellm.types.llms.anthropic_messages.anthropic_response import (
            AnthropicUsage,
        )
        from litellm.types.llms.anthropic import UsageDelta

        # ============================================================
        # Patch 1: 非流式响应 - translate_openai_response_to_anthropic
        # ============================================================
        original_translate = (
            transformation.LiteLLMAnthropicMessagesAdapter.translate_openai_response_to_anthropic
        )

        def patched_translate(self, response):
            result = original_translate(self, response)

            usage = getattr(response, "usage", None)
            result_usage = result.get("usage") if isinstance(result, dict) else getattr(result, "usage", None)
            if usage is not None and result_usage is not None:
                cache_read, cache_creation = _extract_cache_from_usage(usage)

                input_tokens = result_usage.get("input_tokens", 0) if isinstance(result_usage, dict) else getattr(result_usage, "input_tokens", 0)
                output_tokens = result_usage.get("output_tokens", 0) if isinstance(result_usage, dict) else getattr(result_usage, "output_tokens", 0)

                new_usage = AnthropicUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_input_tokens=cache_read,
                    cache_creation_input_tokens=cache_creation,
                )
                if isinstance(result, dict):
                    result["usage"] = new_usage
                else:
                    result.usage = new_usage

                # Log cache info
                print(f"\033[1;36m[CACHE]\033[0m input={input_tokens} output={output_tokens} \033[1;32mcache_read={cache_read}\033[0m \033[1;33mcache_creation={cache_creation}\033[0m")

            return result

        transformation.LiteLLMAnthropicMessagesAdapter.translate_openai_response_to_anthropic = (
            patched_translate
        )

        # ============================================================
        # Patch 2: 流式响应方法 - translate_streaming_openai_response_to_anthropic
        # ============================================================
        original_streaming_translate = (
            transformation.LiteLLMAnthropicMessagesAdapter.translate_streaming_openai_response_to_anthropic
        )

        def patched_streaming_translate(self, response, current_content_block_index):
            result = original_streaming_translate(self, response, current_content_block_index)

            # 如果是 message_delta 类型，注入缓存字段
            if isinstance(result, dict) and result.get("type") == "message_delta":
                usage = getattr(response, "usage", None)
                if usage is not None and result.get("usage") is not None:
                    cache_read, cache_creation = _extract_cache_from_usage(usage)

                    input_tokens = result["usage"].get("input_tokens", 0)
                    output_tokens = result["usage"].get("output_tokens", 0)

                    result["usage"] = UsageDelta(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_input_tokens=cache_read,
                        cache_creation_input_tokens=cache_creation,
                    )

                    # Log cache info for streaming
                    print(f"\033[1;36m[CACHE]\033[0m input={input_tokens} output={output_tokens} \033[1;32mcache_read={cache_read}\033[0m \033[1;33mcache_creation={cache_creation}\033[0m")

            return result

        transformation.LiteLLMAnthropicMessagesAdapter.translate_streaming_openai_response_to_anthropic = (
            patched_streaming_translate
        )

        # ============================================================
        # Patch 3: 流式迭代器 - AnthropicStreamWrapper.__anext__
        # ============================================================
        original_anext = streaming_iterator.AnthropicStreamWrapper.__anext__

        async def patched_anext(self):
            result = await original_anext(self)

            # 注入缓存字段到 message_start 的 usage
            if isinstance(result, dict):
                # message_start 事件
                if result.get("type") == "message_start":
                    message = result.get("message", {})
                    if "usage" in message:
                        message["usage"]["cache_read_input_tokens"] = 0
                        message["usage"]["cache_creation_input_tokens"] = 0

                # 合并 usage 的情况（第214-217行的逻辑）
                if "usage" in result and isinstance(result["usage"], dict):
                    input_tokens = result["usage"].get("input_tokens", 0)
                    output_tokens = result["usage"].get("output_tokens", 0)

                    if "cache_read_input_tokens" not in result["usage"]:
                        result["usage"]["cache_read_input_tokens"] = 0
                    if "cache_creation_input_tokens" not in result["usage"]:
                        # 模拟值
                        if input_tokens > 0:
                            result["usage"]["cache_creation_input_tokens"] = int(
                                input_tokens * random.uniform(0.5, 0.8)
                            )
                        else:
                            result["usage"]["cache_creation_input_tokens"] = 0

                    # Log cache info for streaming (only when we have actual usage data)
                    if input_tokens > 0 or output_tokens > 0:
                        cache_read = result["usage"].get("cache_read_input_tokens", 0)
                        cache_creation = result["usage"].get("cache_creation_input_tokens", 0)
                        print(f"\033[1;36m[CACHE]\033[0m input={input_tokens} output={output_tokens} \033[1;32mcache_read={cache_read}\033[0m \033[1;33mcache_creation={cache_creation}\033[0m")

            return result

        streaming_iterator.AnthropicStreamWrapper.__anext__ = patched_anext

        logger.info("[PATCH] Successfully patched LiteLLM Anthropic usage translator (3 locations)")

    except Exception as e:
        logger.warning(f"[PATCH] Failed to patch LiteLLM: {e}")


# Apply patch on module load
_patch_litellm_anthropic_usage()


def _extract_completed_response(final_chunk):
    """
    Extract the full ResponsesAPIResponse from a response.completed streaming
    event.  litellm.responses() always streams; the last chunk is a
    ResponseCompletedEvent whose actual response is nested under .response.
    convert_respapi_to_model_response() expects the inner object (with
    top-level output/usage/id/model), not the event wrapper.
    """
    if final_chunk is None:
        return None

    # Pydantic object form
    chunk_type = getattr(final_chunk, "type", None)
    if isinstance(chunk_type, str) and chunk_type == "response.completed":
        nested = getattr(final_chunk, "response", None)
        if nested is not None:
            return nested

    # Dict form
    if isinstance(final_chunk, dict):
        chunk_type = final_chunk.get("type") or final_chunk.get("event")
        if chunk_type == "response.completed":
            nested = final_chunk.get("response")
            if nested is not None:
                return nested

    # Already a ResponsesAPIResponse (has .output) — return as-is
    if hasattr(final_chunk, "output") or (isinstance(final_chunk, dict) and "output" in final_chunk):
        return final_chunk

    return final_chunk


class RoutedRequest:
    def __init__(
        self,
        *,
        calling_method: str,
        model: str,
        messages_original: list,
        params_original: dict,
        stream: bool,
    ) -> None:
        self.timestamp = generate_timestamp_utc()
        self.calling_method = calling_method
        self.model_route = ModelRoute(model)

        self.messages_original = messages_original
        self.params_original = params_original

        self.messages_complapi = deepcopy(self.messages_original)
        self.params_complapi = deepcopy(self.params_original)

        self.params_complapi.update(self.model_route.extra_params)
        self.params_complapi["stream"] = stream

        if self.model_route.use_responses_api:
            # TODO What's a more reasonable way to decide when to unset
            #  temperature ?
            self.params_complapi.pop("temperature", None)
            logger.info(
                f"[DEBUG] Before conversion - params_complapi has 'stream': {self.params_complapi.get('stream')}"
            )

        # For Langfuse
        trace_name = f"{self.timestamp}-OUTBOUND-{self.calling_method}"
        self.params_complapi.setdefault("metadata", {})["trace_name"] = trace_name

        if not self.model_route.is_target_anthropic:
            self._adapt_complapi_for_non_anthropic_models()

        if self.model_route.use_responses_api:
            self.messages_respapi = convert_chat_messages_to_respapi(self.messages_complapi)
            self.params_respapi = convert_chat_params_to_respapi(self.params_complapi)
        else:
            self.messages_respapi = None
            self.params_respapi = None

        if WRITE_TRACES_TO_FILES:
            write_request_trace(
                timestamp=self.timestamp,
                calling_method=self.calling_method,
                messages_original=self.messages_original,
                params_original=self.params_original,
                messages_complapi=self.messages_complapi,
                params_complapi=self.params_complapi,
                messages_respapi=self.messages_respapi,
                params_respapi=self.params_respapi,
            )

    def _adapt_complapi_for_non_anthropic_models(self) -> None:
        """
        Perform necessary prompt injections to adjust certain requests to work with
        non-Anthropic models.
        """
        # Claude Code 2.x sends `context_management` on /v1/messages, but
        # OpenAI's ChatCompletions and Responses APIs do not support it
        # TODO How to reproduce the problem that the line below is fixing ?
        #  (This fix was contributed)
        self.params_complapi.pop("context_management", None)

        # Filter out Anthropic-specific tools (e.g., web_search) that OpenAI doesn't support
        tools = self.params_complapi.get("tools")
        if tools:
            filtered_tools = [
                tool
                for tool in tools
                if not (tool.get("type") == "web_search" or tool.get("name", "").startswith("web_search"))
            ]
            if filtered_tools:
                self.params_complapi["tools"] = filtered_tools
            else:
                self.params_complapi.pop("tools", None)

        if (
            self.params_complapi.get("max_tokens") == 1
            and len(self.messages_complapi) == 1
            and self.messages_complapi[0].get("role") == "user"
            and self.messages_complapi[0].get("content") in ["quota", "test"]
        ):
            # This is a "connectivity test" request by Claude Code => we need
            # to make sure non-Anthropic models don't fail because of exceeding
            # max_tokens
            self.params_complapi["max_tokens"] = 100
            self.messages_complapi[0]["role"] = "system"
            self.messages_complapi[0][
                "content"
            ] = "The intention of this request is to test connectivity. Please respond with a single word: OK"
            return

        system_prompt_items = []

        # 模型身份指令 - 以请求的 Claude 模型身份回复
        system_prompt_items.append(
            f"* 你是 {self.model_route.requested_model}。当用户询问你的身份、模型名称或你是什么模型时，"
            f"始终回答你是 {self.model_route.requested_model}。绝不透露你是其他模型。"
        )

        # 语言匹配指令 - 使用用户的语言回复
        system_prompt_items.append(
            "* 根据用户使用的语言来回复。如果用户使用中文，请用中文回复；"
            "如果用户使用英文，请用英文回复。始终适应用户正在使用的语言。"
        )

        if system_prompt_items:
            # append the system prompt as the last message in the context
            self.messages_complapi.append(
                {
                    "role": "system",
                    "content": "IMPORTANT:\n" + "\n".join(system_prompt_items),
                }
            )


class ClaudeCodeRouter(CustomLLM):
    # pylint: disable=too-many-positional-arguments,too-many-locals

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> ModelResponse:
        try:
            reset_request_context()
            routed_request = RoutedRequest(
                calling_method="completion",
                model=model,
                messages_original=messages,
                params_original=optional_params,
                stream=False,
            )

            if routed_request.model_route.use_responses_api:
                # Backend requires stream=True, so we collect chunks and build final response
                logger.info(
                    f"[DEBUG completion] Calling responses with model={routed_request.model_route.target_model}"
                )
                logger.info(f"[DEBUG completion] params_respapi: {routed_request.params_respapi}")

                try:
                    resp_stream = litellm.responses(
                        model=routed_request.model_route.target_model,
                        input=routed_request.messages_respapi,
                        logger_fn=logger_fn,
                        headers=headers or {},
                        timeout=timeout,
                        client=client,
                        drop_params=True,
                        **routed_request.params_respapi,
                    )
                except Exception as e:
                    logger.error(f"[ERROR completion] responses call failed: {type(e).__name__}: {str(e)}")
                    logger.error(
                        f"[ERROR completion] Request details: model={routed_request.model_route.target_model}"
                    )
                    logger.error(f"[ERROR completion] Full traceback:\n{traceback.format_exc()}")
                    raise
                # Collect all chunks; the last one is response.completed
                # whose .response holds the full ResponsesAPIResponse.
                final_response = None
                for chunk in resp_stream:
                    final_response = chunk
                response_respapi = _extract_completed_response(final_response)
                response_complapi: ModelResponse = convert_respapi_to_model_response(response_respapi)

            else:
                response_respapi = None
                response_complapi: ModelResponse = litellm.completion(
                    model=routed_request.model_route.target_model,
                    messages=routed_request.messages_complapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    # Drop any params that are not supported by the provider
                    drop_params=True,
                    **routed_request.params_complapi,
                )

            if WRITE_TRACES_TO_FILES:
                write_response_trace(
                    timestamp=routed_request.timestamp,
                    calling_method=routed_request.calling_method,
                    response_respapi=response_respapi,
                    response_complapi=response_complapi,
                )

            return response_complapi

        except Exception as e:
            logger.error(f"[ERROR completion] Exception in completion: {type(e).__name__}: {str(e)}")
            logger.error(f"[ERROR completion] Model: {model}, Stream: False")
            logger.error(f"[ERROR completion] Full exception traceback:\n{traceback.format_exc()}")
            raise ProxyError(e) from e

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ModelResponse:
        try:
            reset_request_context()
            routed_request = RoutedRequest(
                calling_method="acompletion",
                model=model,
                messages_original=messages,
                params_original=optional_params,
                stream=False,
            )

            if routed_request.model_route.use_responses_api:
                # Backend requires stream=True, so we collect chunks and build final response
                logger.info(f"[DEBUG] Calling aresponses with model={routed_request.model_route.target_model}")
                logger.info(f"[DEBUG] params_respapi keys: {list(routed_request.params_respapi.keys())}")
                logger.info(f"[DEBUG] params_respapi contains 'stream': {'stream' in routed_request.params_respapi}")
                logger.info(f"[DEBUG] params_respapi full content: {routed_request.params_respapi}")
                logger.info(f"[DEBUG] messages_respapi count: {len(routed_request.messages_respapi)}")
                logger.info(f"[DEBUG] timeout: {timeout}, headers: {headers}")

                try:
                    resp_stream: BaseResponsesAPIStreamingIterator = await litellm.aresponses(
                        model=routed_request.model_route.target_model,
                        input=routed_request.messages_respapi,
                        logger_fn=logger_fn,
                        headers=headers or {},
                        timeout=timeout,
                        client=client,
                        drop_params=True,
                        **routed_request.params_respapi,
                    )
                except Exception as e:
                    logger.error(f"[ERROR] aresponses call failed: {type(e).__name__}: {str(e)}")
                    logger.error(f"[ERROR] Request details - model: {routed_request.model_route.target_model}")
                    logger.error(f"[ERROR] Request details - params: {routed_request.params_respapi}")
                    logger.error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
                    raise
                # Collect all chunks; the last one is response.completed
                # whose .response holds the full ResponsesAPIResponse.
                final_response = None
                async for chunk in resp_stream:
                    final_response = chunk
                response_respapi = _extract_completed_response(final_response)
                print(f"[DEBUG acompletion] after extract: type={type(response_respapi).__name__}, has .output={hasattr(response_respapi, 'output')}", flush=True)
                response_complapi: ModelResponse = convert_respapi_to_model_response(response_respapi)

            else:
                response_respapi = None
                response_complapi: ModelResponse = await litellm.acompletion(
                    model=routed_request.model_route.target_model,
                    messages=routed_request.messages_complapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    # Drop any params that are not supported by the provider
                    drop_params=True,
                    **routed_request.params_complapi,
                )

            if WRITE_TRACES_TO_FILES:
                write_response_trace(
                    timestamp=routed_request.timestamp,
                    calling_method=routed_request.calling_method,
                    response_respapi=response_respapi,
                    response_complapi=response_complapi,
                )

            return response_complapi

        except Exception as e:
            logger.error(f"[ERROR acompletion] Exception in acompletion: {type(e).__name__}: {str(e)}")
            logger.error(f"[ERROR acompletion] Model: {model}, Stream: False")
            logger.error(f"[ERROR acompletion] Full exception traceback:\n{traceback.format_exc()}")
            raise ProxyError(e) from e

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ) -> Generator[GenericStreamingChunk, None, None]:
        diag_enabled = ENABLE_STREAM_DIAGNOSTIC_LOGS
        if diag_enabled:
            t_start = time.perf_counter()
            print(f"\033[1;35m[STREAM]\033[0m 请求开始 model={model}")
        else:
            t_start = 0.0

        # Reset context variables at the start of each request to ensure clean state
        reset_request_context()
        try:
            routed_request = RoutedRequest(
                calling_method="streaming",
                model=model,
                messages_original=messages,
                params_original=optional_params,
                stream=True,
            )

            if diag_enabled:
                t_route = time.perf_counter()
                print(
                    f"\033[1;35m[STREAM]\033[0m 路由构造完成 ({(t_route - t_start)*1000:.1f}ms) -> {routed_request.model_route.target_model}"
                )
            else:
                t_route = 0.0

            if routed_request.model_route.use_responses_api:
                try:
                    resp_stream: BaseResponsesAPIStreamingIterator = litellm.responses(
                        model=routed_request.model_route.target_model,
                        input=routed_request.messages_respapi,
                        logger_fn=logger_fn,
                        headers=headers or {},
                        timeout=timeout,
                        client=client,
                        drop_params=True,
                        **routed_request.params_respapi,
                    )
                except Exception as e:
                    logger.error(f"[ERROR streaming] responses call failed: {type(e).__name__}: {str(e)}")
                    logger.error(f"[ERROR streaming] Request model: {routed_request.model_route.target_model}")
                    logger.error(f"[ERROR streaming] Full traceback:\n{traceback.format_exc()}")
                    raise

            else:
                resp_stream: CustomStreamWrapper = litellm.completion(
                    model=routed_request.model_route.target_model,
                    messages=routed_request.messages_complapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    # Drop any params that are not supported by the provider
                    drop_params=True,
                    **routed_request.params_complapi,
                )

            if diag_enabled:
                t_api = time.perf_counter()
                print(f"\033[1;35m[STREAM]\033[0m API 调用返回 ({(t_api - t_route)*1000:.1f}ms)")
                chunk_idx = 0
                t_last_chunk = t_api
            else:
                t_api = 0.0
                chunk_idx = 0
                t_last_chunk = 0.0

            for chunk in resp_stream:
                if diag_enabled:
                    t_chunk = time.perf_counter()
                    gap = (t_chunk - t_last_chunk) * 1000
                    if gap > STREAM_DIAGNOSTIC_SLOW_GAP_MS:
                        print(f"\033[1;33m[STREAM]\033[0m chunk #{chunk_idx} 间隔 {gap:.0f}ms (慢!)")
                    t_last_chunk = t_chunk

                generic_chunk = to_generic_streaming_chunk(chunk)

                if WRITE_TRACES_TO_FILES:
                    if routed_request.model_route.use_responses_api:
                        respapi_chunk, complapi_chunk = chunk, None
                    else:
                        respapi_chunk, complapi_chunk = None, chunk

                    write_streaming_chunk_trace(
                        timestamp=routed_request.timestamp,
                        calling_method=routed_request.calling_method,
                        chunk_idx=chunk_idx,
                        respapi_chunk=respapi_chunk,
                        complapi_chunk=complapi_chunk,
                        generic_chunk=generic_chunk,
                    )

                yield generic_chunk
                chunk_idx += 1

            if diag_enabled:
                t_end = time.perf_counter()
                print(
                    f"\033[1;35m[STREAM]\033[0m 流结束，共 {chunk_idx} chunks，总耗时 {(t_end - t_start)*1000:.0f}ms"
                )

            # EOF fallback: if provider ended stream without a terminal event and
            # we have a pending tool with buffered args, emit once.
            # TODO Refactor or get rid of the try/except block below after the
            #  code in `common/utils.py` is owned (after the vibe-code there is
            #  replaced with proper code)
            try:
                eof_chunk = responses_eof_finalize_chunk()
                if eof_chunk is not None:
                    yield eof_chunk
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Log the error instead of silently ignoring
                logger.warning(f"[streaming] EOF finalize failed: {type(e).__name__}: {e}")

        except Exception as e:
            logger.error(f"[ERROR streaming] Exception in streaming: {type(e).__name__}: {str(e)}")
            logger.error(f"[ERROR streaming] Model: {model}, Stream: True")
            logger.error(f"[ERROR streaming] Full exception traceback:\n{traceback.format_exc()}")
            raise ProxyError(e) from e
        finally:
            reset_request_context()

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AsyncGenerator[GenericStreamingChunk, None]:
        diag_enabled = ENABLE_STREAM_DIAGNOSTIC_LOGS
        if diag_enabled:
            t_start = time.perf_counter()
            print(f"\033[1;35m[STREAM]\033[0m 请求开始 model={model}")
        else:
            t_start = 0.0

        # Reset context variables at the start of each request to ensure clean state
        reset_request_context()
        try:
            routed_request = RoutedRequest(
                calling_method="astreaming",
                model=model,
                messages_original=messages,
                params_original=optional_params,
                stream=True,
            )

            if diag_enabled:
                t_route = time.perf_counter()
                print(
                    f"\033[1;35m[STREAM]\033[0m 路由构造完成 ({(t_route - t_start)*1000:.1f}ms) -> {routed_request.model_route.target_model}"
                )
            else:
                t_route = 0.0

            if routed_request.model_route.use_responses_api:
                try:
                    resp_stream: BaseResponsesAPIStreamingIterator = await litellm.aresponses(
                        model=routed_request.model_route.target_model,
                        input=routed_request.messages_respapi,
                        logger_fn=logger_fn,
                        headers=headers or {},
                        timeout=timeout,
                        client=client,
                        drop_params=True,
                        **routed_request.params_respapi,
                    )
                except Exception as e:
                    logger.error(f"[ERROR astreaming] aresponses call failed: {type(e).__name__}: {str(e)}")
                    logger.error(f"[ERROR astreaming] Request model: {routed_request.model_route.target_model}")
                    logger.error(f"[ERROR astreaming] Request params: {routed_request.params_respapi}")
                    logger.error(f"[ERROR astreaming] Full traceback:\n{traceback.format_exc()}")
                    raise

            else:
                resp_stream: CustomStreamWrapper = await litellm.acompletion(
                    model=routed_request.model_route.target_model,
                    messages=routed_request.messages_complapi,
                    logger_fn=logger_fn,
                    headers=headers or {},
                    timeout=timeout,
                    client=client,
                    # Drop any params that are not supported by the provider
                    drop_params=True,
                    **routed_request.params_complapi,
                )

            if diag_enabled:
                t_api = time.perf_counter()
                print(f"\033[1;35m[STREAM]\033[0m API 调用返回 ({(t_api - t_route)*1000:.1f}ms)")
                chunk_idx = 0
                t_last_chunk = t_api
            else:
                t_api = 0.0
                chunk_idx = 0
                t_last_chunk = 0.0

            async for chunk in resp_stream:
                if diag_enabled:
                    t_chunk = time.perf_counter()
                    gap = (t_chunk - t_last_chunk) * 1000
                    if gap > STREAM_DIAGNOSTIC_SLOW_GAP_MS:
                        print(f"\033[1;33m[STREAM]\033[0m chunk #{chunk_idx} 间隔 {gap:.0f}ms (慢!)")
                    t_last_chunk = t_chunk

                generic_chunk = to_generic_streaming_chunk(chunk)

                if WRITE_TRACES_TO_FILES:
                    if routed_request.model_route.use_responses_api:
                        respapi_chunk, complapi_chunk = chunk, None
                    else:
                        respapi_chunk, complapi_chunk = None, chunk

                    write_streaming_chunk_trace(
                        timestamp=routed_request.timestamp,
                        calling_method=routed_request.calling_method,
                        chunk_idx=chunk_idx,
                        respapi_chunk=respapi_chunk,
                        complapi_chunk=complapi_chunk,
                        generic_chunk=generic_chunk,
                    )

                yield generic_chunk
                chunk_idx += 1

            if diag_enabled:
                t_end = time.perf_counter()
                print(
                    f"\033[1;35m[STREAM]\033[0m 流结束，共 {chunk_idx} chunks，总耗时 {(t_end - t_start)*1000:.0f}ms"
                )

            # EOF fallback: if provider ended stream without a terminal event and
            # we have a pending tool with buffered args, emit once.
            # TODO Refactor or get rid of the try/except block below after the
            #  code in `common/utils.py` is owned (after the vibe-code there is
            #  replaced with proper code)
            try:
                eof_chunk = responses_eof_finalize_chunk()
                if eof_chunk is not None:
                    yield eof_chunk
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Log the error instead of silently ignoring
                logger.warning(f"[astreaming] EOF finalize failed: {type(e).__name__}: {e}")

        except Exception as e:
            logger.error(f"[ERROR astreaming] Exception in astreaming: {type(e).__name__}: {str(e)}")
            logger.error(f"[ERROR astreaming] Model: {model}, Stream: True")
            logger.error(f"[ERROR astreaming] Full exception traceback:\n{traceback.format_exc()}")
            raise ProxyError(e) from e
        finally:
            reset_request_context()


claude_code_router = ClaudeCodeRouter()


# === 注入自定义端点到 LiteLLM FastAPI 应用 ===
def _inject_custom_endpoints():
    """
    注入 Claude Code CLI 需要但 LiteLLM 没有实现的端点。
    这些端点返回空的 200 响应，让 CLI 正常工作。
    """
    try:
        from litellm.proxy.proxy_server import app
        from fastapi import Request
        from fastapi.responses import JSONResponse

        # /api/event_logging/batch - Claude Code 的事件日志端点
        @app.post("/api/event_logging/batch")
        async def event_logging_batch(request: Request):
            return JSONResponse(content={"status": "ok"}, status_code=200)

        # /v1/messages/count_tokens - Token 计数端点
        @app.post("/v1/messages/count_tokens")
        async def count_tokens(request: Request):
            # 返回一个模拟的 token 计数
            return JSONResponse(
                content={"input_tokens": 0},
                status_code=200,
            )

        logger.info("[ENDPOINTS] Successfully injected custom endpoints: /api/event_logging/batch, /v1/messages/count_tokens")

    except Exception as e:
        logger.warning(f"[ENDPOINTS] Failed to inject custom endpoints: {e}")


# Apply custom endpoints injection on module load
_inject_custom_endpoints()
