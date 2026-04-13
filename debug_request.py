#!/usr/bin/env python3
"""Debug script to inspect raw streaming responses from vLLM.

Reproduces the exact bug path in trace_replay_tester.py:
  1. Send a user message, collect the response (including <think> text)
  2. Append that response as an assistant message (as the replayer does
     due to the store_assistant_response guard being dead code)
  3. Send the conversation again with the assistant message appended
     (simulating the paired n-type request)

If Request 2 returns 0 output tokens, it confirms the root cause.

Usage:
    python3 debug_request.py [--endpoint http://localhost:8888] [--max-tokens 120]
"""

import asyncio
import argparse
import openai


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8888")
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    base_url = args.endpoint.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    client = openai.AsyncOpenAI(api_key="EMPTY", base_url=base_url)

    # Auto-detect model (same as replayer)
    models = await client.models.list()
    model = models.data[0].id
    print(f"Model: {model}\n")

    messages = [{"role": "user", "content": "Write a short Python function to reverse a string."}]

    # Build params exactly like trace_replay_tester._build_request_params
    params = {
        "model": model,
        "messages": list(messages),
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # === Request 1: normal user message ===
    print(f"=== Request 1 (user message only) ===")
    print(f"max_tokens={args.max_tokens}, stream=True")
    print(f"messages: [user]")
    print()

    response_text, token_count = await send_and_dump(client, params)

    # === Request 2: what the replayer ACTUALLY sends for the paired request ===
    # Due to the bug, store_assistant_response() stores the response immediately
    # (the 'non_streaming' guard never matches because normalize_request sets
    # all types to 'streaming'). So the paired request gets sent with the
    # assistant response already appended to the conversation.
    print(f"\n=== Request 2 (with assistant response appended — reproduces bug) ===")
    print(f"messages: [user, assistant]")
    print(f"assistant content (first 200 chars): {response_text[:200]!r}")
    print()

    bug_messages = list(messages) + [{"role": "assistant", "content": response_text}]
    bug_params = dict(params, messages=bug_messages)
    await send_and_dump(client, bug_params)

    # === Request 3: what it SHOULD send (identical messages, no assistant) ===
    print(f"\n=== Request 3 (identical user-only messages — correct behavior) ===")
    print(f"messages: [user]")
    print()

    await send_and_dump(client, params)


async def send_and_dump(client, params):
    response = await client.chat.completions.create(**params)

    chunk_idx = 0
    token_count = 0
    content_chars = 0
    reasoning_chars = 0
    response_text = ""

    async for chunk in response:
        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta

            # Replicate the exact counting logic from trace_replay_tester
            content_text = delta.content or ""
            reasoning_text = getattr(delta, "reasoning_content", None) or ""
            chunk_text = content_text or reasoning_text

            if chunk_text:
                token_count += 1  # simplified; real code uses tokenizer
                content_chars += len(content_text)
                reasoning_chars += len(reasoning_text)
                if content_text:
                    response_text += content_text

            # Print finish_reason when it appears
            if choice.finish_reason:
                print(f"  finish_reason: {choice.finish_reason!r} (at chunk {chunk_idx})")

        if chunk.usage:
            print(f"  usage: {chunk.usage}")

        chunk_idx += 1

    print(f"\n--- Summary ---")
    print(f"  Total chunks: {chunk_idx}")
    print(f"  Chunks with text (would be counted): {token_count}")
    print(f"  Content chars: {content_chars}")
    print(f"  Reasoning chars: {reasoning_chars}")

    return response_text, token_count


if __name__ == "__main__":
    asyncio.run(main())
