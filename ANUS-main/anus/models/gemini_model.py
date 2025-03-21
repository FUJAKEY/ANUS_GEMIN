"""
Gemini Model module for using the Gemini AI model.
"""

from typing import Dict, Any, Iterable
import logging

from google import genai
from google.genai import types
from anus.models.base.base_model import BaseModel

class GeminiModel(BaseModel):
    """
    Gemini Model implementation using the Google GenAI API.
    Реализует функционал, аналогичный OpenAIModel: генерация текста, стриминг и чат.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = "", temperature: float = 0.0, **kwargs):
        """
        Initialize the GeminiModel.
        
        Args:
            model_name: Имя модели Gemini (по умолчанию "gemini-2.0-flash").
            api_key: API key для доступа к Gemini.
            temperature: Параметр температуры для генерации.
            kwargs: Дополнительные параметры конфигурации.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logging.error(f"Failed to initialize Gemini client: {e}")
            raise e

    def generate_text(self, contents: Any) -> str:
        """
        Генерирует текст на основе переданных данных.
        
        Args:
            contents: Текстовый запрос или список, содержащий изображение и текстовый запрос.
        
        Returns:
            Сгенерированный текст.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(temperature=self.temperature)
            )
            return response.text
        except Exception as e:
            logging.error(f"Error generating content with Gemini: {e}")
            return ""
    
    def generate_text_stream(self, contents: Any) -> Iterable[str]:
        """
        Генерирует текст в стриминговом режиме.
        
        Args:
            contents: Текстовый запрос или список входов.
        
        Returns:
            Итератор, возвращающий порции сгенерированного текста.
        """
        try:
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(temperature=self.temperature)
            )
            for chunk in response:
                yield chunk.text
        except Exception as e:
            logging.error(f"Error in streaming generation with Gemini: {e}")
            return iter([])

    def create_chat(self) -> "GeminiChat":
        """
        Создает чат-сессию для диалоговых задач.
        
        Returns:
            Объект GeminiChat для дальнейшего общения.
        """
        try:
            chat_session = self.client.chats.create(model=self.model_name)
            return GeminiChat(chat_session)
        except Exception as e:
            logging.error(f"Error creating chat session with Gemini: {e}")
            raise e

    def get_model_details(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели.
        
        Returns:
            Словарь с деталями модели.
        """
        return {
            "provider": "gemini",
            "model_name": self.model_name,
            "temperature": self.temperature
        }

class GeminiChat:
    """
    Обёртка для чат-сессии Gemini, предоставляющая методы для отправки сообщений и получения истории.
    """
    def __init__(self, chat_session):
        self.chat_session = chat_session

    def send_message(self, message: str) -> str:
        """
        Отправляет сообщение в чат-сессию.
        
        Args:
            message: Сообщение для отправки.
        
        Returns:
            Ответ от модели в виде текста.
        """
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            logging.error(f"Error sending message in Gemini chat: {e}")
            return ""

    def get_history(self) -> Iterable[Dict[str, Any]]:
        """
        Получает историю сообщений из чат-сессии.
        
        Returns:
            Итератор по сообщениям чата, где каждое сообщение представлено в виде словаря.
        """
        try:
            history = self.chat_session.get_history()
            # Преобразуем объекты сообщений в словари для удобства
            return [{"role": message.role, "text": message.parts[0].text} for message in history]
        except Exception as e:
            logging.error(f"Error retrieving Gemini chat history: {e}")
            return iter([])
