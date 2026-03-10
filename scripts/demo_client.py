"""Demo WebSocket client — sends a WAV file and prints streaming responses.

Usage
=====
    python scripts/demo_client.py --audio data/samples/speech_sim_2s.wav
    python scripts/demo_client.py --audio data/samples/tone_440hz_1s.wav --image data/samples/frame.jpg
    python scripts/demo_client.py --text "What is the weather like?"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_demo(
    audio_path: str | None = None,
    image_path: str | None = None,
    text: str | None = None,
    host: str = "localhost",
    port: int = 8000,
) -> None:
    try:
        import websockets
    except ImportError:
        print("Please install websockets: pip install websockets")
        return

    uri = f"ws://{host}:{port}/ws/stream"
    print(f"\nConnecting to {uri} ...\n")

    async with websockets.connect(uri) as ws:
        # Wait for ready
        msg = json.loads(await ws.recv())
        print(f"Server: {msg}")

        if audio_path:
            audio_bytes = Path(audio_path).read_bytes()
            encoded = base64.b64encode(audio_bytes).decode()
            await ws.send(json.dumps({"type": "audio_chunk", "data": encoded, "sample_rate": 16000}))
            print(f"Sent audio: {audio_path} ({len(audio_bytes):,} bytes)")
        elif text:
            # For text-only demo, we'd need a REST endpoint
            print("Text mode: use POST /api/v1/stream/query instead")
            return

        if image_path:
            img_bytes = Path(image_path).read_bytes()
            encoded_img = base64.b64encode(img_bytes).decode()
            await ws.send(json.dumps({"type": "image_frame", "data": encoded_img}))
            print(f"Sent image: {image_path} ({len(img_bytes):,} bytes)")

        await ws.send(json.dumps({"type": "end_of_speech"}))
        print("\n--- Streaming response ---")

        response_text = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "transcript":
                print(f"\nYou said: \"{event.get('text')}\"")
            elif event_type == "llm_token":
                print(event.get("token", ""), end="", flush=True)
                response_text.append(event.get("token", ""))
            elif event_type == "tts_chunk":
                data_len = len(event.get("data", ""))
                print(f"\n[Audio chunk: {data_len} base64 chars]")
            elif event_type == "latency_report":
                print(f"\n\n--- Latency Report ---")
                print(f"  Total:  {event.get('total_actual_ms', 0):.0f}ms / {event.get('total_budget_ms', 0)}ms budget")
                print(f"  Within budget: {event.get('within_budget')}")
                overruns = event.get("over_budget_stages", [])
                if overruns:
                    print(f"  Over-budget stages: {', '.join(overruns)}")
                break
            elif event_type == "error":
                print(f"\nError: {event.get('message')}")
                break


def main():
    parser = argparse.ArgumentParser(description="Demo WebSocket client")
    parser.add_argument("--audio", help="Path to WAV file")
    parser.add_argument("--image", help="Path to JPEG image")
    parser.add_argument("--text", help="Text query (uses REST fallback)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not args.audio and not args.text:
        parser.error("Provide --audio or --text")

    asyncio.run(run_demo(args.audio, args.image, args.text, args.host, args.port))


if __name__ == "__main__":
    main()
