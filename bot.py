import os
import redis
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from dotenv import load_dotenv
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import io

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380)) 
MODEL_PATH = os.getenv("MODEL_PATH", "CompVis/stable-diffusion-v1-4")

redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_PATH)
pipeline.to(device)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

app = FastAPI()

INSTRUCTION = (
    "\U0001F44B Привет! Я бот для улучшения фотографий. Вот как я работаю:\n\n"
    "1. Отправьте мне фотографию, которую хотите улучшить.\n"
    "2. Я обработаю изображение и отправлю вам улучшенный результат!"
)

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply(INSTRUCTION, parse_mode='Markdown')

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]  
    file = await bot.get_file(photo.file_id)
    file_path = file.file_path
    file_bytes = await bot.download_file(file_path)

    input_image = Image.open(io.BytesIO(file_bytes.read()))

    try:
        enhanced_image = process_image(input_image)
        output_buffer = io.BytesIO()
        enhanced_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        await bot.send_photo(
            chat_id=message.chat.id, photo=output_buffer, caption="Ваше улучшенное изображение!"
        )
    except Exception as e:
        await message.reply("Произошла ошибка при обработке изображения. Пожалуйста, попробуйте снова.")
        print(f"Error processing image: {e}")

def process_image(input_image: Image.Image) -> Image.Image:
    try:
        init_image = input_image.convert("RGB")
        init_image = init_image.resize((512, 512))
        
        result = pipeline(
            init_image=init_image,
            strength=0.75,
            guidance_scale=7.5
        ).images[0]
        return result
    except Exception as e:
        raise RuntimeError(f"Ошибка в процессе обработки изображения: {e}")

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
