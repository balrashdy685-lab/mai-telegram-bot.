import os
import logging
import asyncio
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from collections import defaultdict

import aiohttp
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============== الإعدادات ==============
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = "meta-llama/llama-3.1-405b-instruct:free"
GEMINI_MODEL = "gemini-2.0-flash"
MAX_HISTORY = 20

SYSTEM_PROMPT = """أنت مساعد ذكي في بوت تلجرام. اتبع هذه القواعد بدقة:
1. نفّذ الطلب مباشرة بدون مقدمات.
2. نسّق ردودك بـ Markdown يتوافق مع تلجرام.
3. اجعل الردود مختصرة ومباشرة.
"""

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
user_history = defaultdict(list)

# ============== دوال الاتصال بالـ APIs ==============
async def call_gemini(messages: list) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    contents = []
    for m in messages:
        if m["role"] == "system": continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload = {"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=60) as resp:
            data = await resp.json()
            if resp.status != 200: raise RuntimeError(f"Gemini error: {data}")
            return data["candidates"][0]["content"]["parts"][0]["text"]

async def call_openrouter(messages: list) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENROUTER_MODEL, "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=60) as resp:
            data = await resp.json()
            if resp.status != 200: raise RuntimeError(f"OpenRouter error: {data}")
            return data["choices"][0]["message"]["content"]

async def get_ai_reply(messages: list) -> str:
    try:
        return await call_gemini(messages)
    except Exception as e:
        logger.warning(f"Gemini failed, switching to OpenRouter: {e}")
        try:
            return await call_openrouter(messages)
        except Exception as e2:
            logger.error(f"OpenRouter also failed: {e2}")
            return "⚠️ عذراً، حدث خطأ في الاتصال بخدمات الذكاء الاصطناعي."

def to_telegram_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"```(\w+)?\n(.*?)```", lambda m: f"<pre><code>{m.group(2)}</code></pre>", text, flags=re.DOTALL)
    return text

async def send_long_message(update: Update, text: str):
    formatted = to_telegram_markdown(text)
    try:
        await update.message.reply_text(formatted, parse_mode=ParseMode.HTML)
    except Exception:
        await update.message.reply_text(re.sub(r"<[^>]+>", "", formatted))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    history = user_history[user_id]
    history.append({"role": "user", "content": user_text})
    history[:] = history[-MAX_HISTORY:]
    reply = await get_ai_reply(history)
    history.append({"role": "assistant", "content": reply})
    history[:] = history[-MAX_HISTORY:]
    await send_long_message(update, reply)

# ============== السيرفر الصحي لإرضاء Render ==============
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, format, *args):
        pass

def run_http_server():
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"🩺 السيرفر الصحي شغّال على المنفذ {port}")
    server.serve_forever()

# ============== التشغيل الأساسي ==============
def main():
    # تشغيل السيرفر الصحي في خيط مستقل تماماً لكي لا يتعارض مع البوت
    server_thread = Thread(target=run_http_server, daemon=True)
    server_thread.start()

    # بناء وتشغيل بوت التيليجرام
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("🚀 البوت شغّال...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
