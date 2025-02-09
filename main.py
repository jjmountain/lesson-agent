import logging
import asyncio
from typing import List, Optional, AsyncGenerator

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero, turn_detector, elevenlabs
from livekit.plugins.openai import stt

load_dotenv(dotenv_path=".env")
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

groq_stt = stt.STT.with_groq(
  model="whisper-large-v3-turbo",
)

eleven_tts=elevenlabs.tts.TTS(
    model="eleven_turbo_v2_5",
    voice=elevenlabs.tts.Voice(
        id="GKDaBI8TKSBJVhsCLD6n",
        name="Asahi",
        category="premade",
        settings=elevenlabs.tts.VoiceSettings(
            stability=0.71,
            similarity_boost=0.5,
            style=0.0,
            use_speaker_boost=True
        ),
    ),
    language="ja",
    streaming_latency=3,
    enable_ssml_parsing=False,
    chunk_length_schedule=[80, 120, 200, 260],
)

# 1. Custom LLM Bridge (Dummy implementation)
class CustomLLM(llm.LLM):
    async def chat(self, chat_ctx: llm.ChatContext, fnc_ctx: Optional[llm.FunctionContext] = None) -> AsyncGenerator[str, None]:
        # Let's inspect what chat_ctx contains
        logger.info("Chat Context Messages:")
        for msg in chat_ctx.messages:
            logger.info(f"Role: {msg.role}, Text: {msg.content}")
            
        # Convert to messages format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in chat_ctx.messages
        ]
        
        logger.info(f"Converted Messages: {messages}")
        
        # Dummy response for now
        response = "Hello! I am a mock Japanese teaching assistant. どうぞよろしく！"
        for chunk in response.split():
            yield chunk + " "

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    # 3. Create Agent with Custom LLM
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=groq_stt,
        llm=CustomLLM(),  # Your custom bridge
        tts=eleven_tts,
        turn_detector=turn_detector.EOUModel(),
        chat_ctx=llm.ChatContext().append(
            text="You are a Japanese language teaching assistant.",
            role="system"
        )
    )

    # 4. Manual Interaction Handling
    @agent.on("llm_response")
    def on_llm_response(response: str):
        logger.info(f"LLM Response: {response}")
        # Create task for async operations
        asyncio.create_task(agent.say(response))

    agent.start(ctx.room, participant)
    await agent.say("Hi richard, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
