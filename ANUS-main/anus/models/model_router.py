"""
Model Router module for dynamic model selection.
"""

from typing import Dict, List, Any, Optional, Union, Type
import logging

from anus.models.base.base_model import BaseModel
from anus.models.openai_model import OpenAIModel
from anus.models.gemini_model import GeminiModel  # Импорт нового класса модели

class ModelRouter:
    """
    Router for dynamically selecting and managing language models.
    
    Provides functionality for:
    - Registering different model implementations
    - Selecting models based on task requirements
    - Fallback mechanisms for reliability
    """
    
    def __init__(self, default_model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize a ModelRouter instance.
        
        Args:
            default_model_config: Configuration for the default model.
        """
        self.models: Dict[str, BaseModel] = {}
        # Регистрация классов моделей для разных провайдеров
        self.model_classes: Dict[str, Type[BaseModel]] = {
            "openai": OpenAIModel,
            "gemini": GeminiModel
        }
        # Если конфигурация по умолчанию не передана, используем openai
        self.default_model_config = default_model_config or {
            "provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.0
        }
        self.default_model = None
    
    def register_model(self, name: str, model: BaseModel) -> None:
        """
        Register a model instance.
        
        Args:
            name: A unique name for the model.
            model: The model instance to register.
        """
        self.models[name] = model
        logging.info(f"Registered model: {name}")
    
    def register_model_class(self, provider: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class for a provider.
        
        Args:
            provider: The model provider name.
            model_class: The model class to register.
        """
        self.model_classes[provider] = model_class
        logging.info(f"Registered model class for provider: {provider}")
    
    def get_model(self, name_or_config: Union[str, Dict[str, Any]]) -> BaseModel:
        """
        Get a model instance by name or create one from config.
        
        Args:
            name_or_config: Either a model name or a model configuration dictionary.
            
        Returns:
            A model instance.
        """
        # Если передана строка, ищем по имени
        if isinstance(name_or_config, str):
            if name_or_config in self.models:
                return self.models[name_or_config]
            logging.warning(f"Model '{name_or_config}' not found. Using default model.")
            return self.get_default_model()
        
        # Если передана конфигурация, создаем модель по ней
        elif isinstance(name_or_config, dict):
            return self._create_model_from_config(name_or_config)
        
        # Неверный тип входных данных
        else:
            logging.error(f"Invalid model specification: {name_or_config}")
            return self.get_default_model()
    
    def get_default_model(self) -> BaseModel:
        """
        Get the default model, creating it if necessary.
        
        Returns:
            The default model instance.
        """
        if self.default_model is None:
            self.default_model = self._create_model_from_config(self.default_model_config)
        return self.default_model
    
    def select_model_for_task(self, task: str, requirements: Dict[str, Any] = None) -> BaseModel:
        """
        Select an appropriate model for a given task.
        
        Args:
            task: The task description.
            requirements: Optional requirements for the model.
            
        Returns:
            The selected model instance.
        """
        # Если заданы требования, создаем модель с нужной конфигурацией
        if requirements:
            return self._create_model_from_config(requirements)
        return self.get_default_model()
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance from a configuration dictionary.
        
        Args:
            config: The model configuration.
            
        Returns:
            A model instance.
        """
        # Определяем провайдера (по умолчанию openai)
        provider = config.get("provider", "openai").lower()
        
        # Если провайдер неизвестен, логируем ошибку и используем openai
        if provider not in self.model_classes:
            logging.error(f"Unknown model provider: {provider}. Using OpenAI as fallback.")
            provider = "openai"
        
        # Дополнительная проверка для Gemini: необходим api_key
        if provider == "gemini" and "api_key" not in config:
            logging.error("API key must be provided for gemini provider.")
        
        try:
            model_class = self.model_classes[provider]
            # Подготовка параметров для создания модели
            kwargs = config.copy()
            kwargs.pop("provider", None)
            return model_class(**kwargs)
            
        except Exception as e:
            logging.error(f"Error creating model for provider {provider}: {e}")
            try:
                return OpenAIModel(model_name="gpt-4")
            except Exception:
                raise ValueError(f"Failed to create model: {e}")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            A list of model information dictionaries.
        """
        models_info = []
        # Добавляем зарегистрированные экземпляры моделей
        for name, model in self.models.items():
            info = {
                "name": name,
                "type": type(model).__name__,
                "model_name": model.model_name,
                "details": model.get_model_details()
            }
            models_info.append(info)
        
        # Добавляем доступных провайдеров, если их еще нет среди зарегистрированных моделей
        for provider, model_class in self.model_classes.items():
            if provider not in [info["details"].get("provider") for info in models_info]:
                models_info.append({
                    "name": provider,
                    "type": model_class.__name__,
                    "details": {"provider": provider}
                })
        
        return models_info
