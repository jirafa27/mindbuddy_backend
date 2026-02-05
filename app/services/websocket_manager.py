"""Менеджер для управления WebSocket соединениями и подпиской на RabbitMQ"""
import json
import logging
from typing import Dict
from fastapi import WebSocket
from aio_pika import connect_robust, IncomingMessage
from aio_pika.abc import AbstractConnection, AbstractChannel, AbstractQueue

from app.core.config import settings

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Менеджер для управления WebSocket соединениями и подпиской на RabbitMQ"""
    
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}  # user_id -> WebSocket
        self.rabbitmq_connection: AbstractConnection = None
        self.rabbitmq_channel: AbstractChannel = None
        self.rabbitmq_queue: AbstractQueue = None
        self._consumer_tags: Dict[int, str] = {}  # user_id -> consumer_tag
        
    async def connect(self):
        """Подключается к RabbitMQ"""
        try:
            self.rabbitmq_connection = await connect_robust(settings.RABBITMQ_URL)
            self.rabbitmq_channel = await self.rabbitmq_connection.channel()
            
            # Объявляем очередь
            self.rabbitmq_queue = await self.rabbitmq_channel.declare_queue(
                "watcher_tasks",
                durable=True
            )
            
            logger.info("WebSocketManager connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self):
        """Отключается от RabbitMQ. Устойчив к уже закрытому соединению (shutdown брокера)."""
        consumer_tags = list(self._consumer_tags.values())
        self._consumer_tags.clear()
        queue, channel, connection = self.rabbitmq_queue, self.rabbitmq_channel, self.rabbitmq_connection
        self.rabbitmq_queue = self.rabbitmq_channel = self.rabbitmq_connection = None

        if queue and consumer_tags:
            for consumer_tag in consumer_tags:
                try:
                    await queue.cancel(consumer_tag)
                except Exception:
                    pass
        if channel:
            try:
                await channel.close()
            except Exception:
                pass
        # Закрываем соединение первым — останавливает фоновый reconnection у connect_robust
        if connection:
            try:
                await connection.close()
            except Exception:
                pass

        logger.info("WebSocketManager disconnected from RabbitMQ")
    
    async def connect_websocket(self, websocket: WebSocket, user_id: int):
        """Подключает WebSocket и подписывается на задачи для пользователя"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        
        # Создаём потребителя для этого пользователя (если RabbitMQ подключён)
        if self.rabbitmq_queue:
            await self._start_consumer(user_id)
        else:
            logger.warning(f"RabbitMQ not connected, WebSocket for user {user_id} will not receive tasks")
        
        logger.info(f"WebSocket connected for user {user_id}")
    
    async def disconnect_websocket(self, user_id: int):
        """Отключает WebSocket и останавливает потребителя"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].close()
            except Exception:
                pass
            del self.active_connections[user_id]
        
        # Останавливаем потребителя
        if user_id in self._consumer_tags and self.rabbitmq_queue:
            try:
                await self.rabbitmq_queue.cancel(self._consumer_tags[user_id])
            except Exception as e:
                logger.warning(f"Failed to cancel consumer for user {user_id}: {e}")
            del self._consumer_tags[user_id]
        
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def _start_consumer(self, user_id: int):
        """Запускает потребителя RabbitMQ для конкретного пользователя"""
        if user_id in self._consumer_tags:
            return  # Потребитель уже запущен
        
        if not self.rabbitmq_queue:
            logger.error(f"Cannot start consumer for user {user_id}: RabbitMQ queue not connected")
            return
        
        async def process_message(message: IncomingMessage):
            """Обрабатывает сообщение из RabbitMQ"""
            try:
                task_data = json.loads(message.body.decode())
                
                # Проверяем, что задача для этого пользователя
                if task_data.get("user_id") != user_id:
                    # Не наша задача, возвращаем в очередь без подтверждения
                    await message.nack(requeue=True)
                    return
                
                # Задача для нашего пользователя
                # Отправляем задачу через WebSocket
                if user_id in self.active_connections:
                    websocket = self.active_connections[user_id]
                    try:
                        await websocket.send_json(task_data)
                        # Подтверждаем получение сообщения только после успешной отправки
                        await message.ack()
                        logger.info(f"Task sent to user {user_id} via WebSocket")
                    except Exception as e:
                        logger.error(f"Failed to send task via WebSocket to user {user_id}: {e}")
                        # Если не удалось отправить, возвращаем в очередь
                        await message.nack(requeue=True)
                else:
                    # WebSocket отключён, возвращаем в очередь
                    await message.nack(requeue=True)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")
                await message.nack(requeue=False)  # Не возвращаем некорректное сообщение
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await message.nack(requeue=True)
        
        # Создаём потребителя (consume() возвращает consumer_tag — строку)
        consumer_tag = await self.rabbitmq_queue.consume(process_message)
        self._consumer_tags[user_id] = consumer_tag
        
        logger.info(f"Consumer started for user {user_id} with tag {consumer_tag}")
    
    async def send_personal_message(self, user_id: int, message: dict):
        """Отправляет персональное сообщение пользователю через WebSocket"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json(message)
                return True
            except Exception as e:
                logger.error(f"Failed to send message to user {user_id}: {e}")
                return False
        return False
    
    def is_connected(self, user_id: int) -> bool:
        """Проверяет, подключён ли пользователь"""
        return user_id in self.active_connections
