import json
import logging
from typing import Optional
import pika
from pika.exceptions import AMQPConnectionError

from app.core.config import settings

logger = logging.getLogger(__name__)


class RabbitMQService:
    def __init__(self):
        self.connection_params = pika.URLParameters(settings.RABBITMQ_URL)

    def _get_connection(self):
        try:
            return pika.BlockingConnection(self.connection_params)
        except AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise Exception(f"Failed to connect to RabbitMQ: {str(e)}")

    def send_message(
        self,
        queue_name: str,
        message: dict,
        persistent: bool = True,
    ) -> bool:
        connection = None
        try:
            connection = self._get_connection()
            channel = connection.channel()
            channel.queue_declare(queue=queue_name, durable=True)
            body = json.dumps(message, ensure_ascii=False)
            channel.basic_publish(
                exchange="",
                routing_key=queue_name,
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=2 if persistent else 1,
                    content_type="application/json",
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send message to RabbitMQ: {e}")
            raise Exception(f"Failed to send message to RabbitMQ: {str(e)}")
        finally:
            if connection and not connection.is_closed:
                connection.close()

    def send_watcher_task(
        self,
        file_id: int,
        user_id: int,
        filename: str,
        file_type: str,
        file_size: int,
        download_url: str,
        local_path: Optional[str] = None,
    ) -> bool:
        message = {
            "action": "download_and_sync",
            "file_id": file_id,
            "user_id": user_id,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size,
            "download_url": download_url,
            "local_path": local_path,
            "status": "pending",
        }
        return self.send_message(queue_name="watcher_tasks", message=message, persistent=True)

    def get_watcher_task(self, user_id: int, timeout: int = 1) -> Optional[dict]:
        connection = None
        try:
            connection = self._get_connection()
            channel = connection.channel()
            channel.queue_declare(queue="watcher_tasks", durable=True)
            method_frame, header_frame, body = channel.basic_get(queue="watcher_tasks", auto_ack=False)
            if method_frame is None:
                return None
            try:
                message = json.loads(body)
                if message.get("user_id") != user_id:
                    channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
                    return None
                channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                return message
            except json.JSONDecodeError:
                channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)
                return None
        except Exception as e:
            logger.error(f"Failed to get task from RabbitMQ: {e}")
            raise Exception(f"Failed to get task from RabbitMQ: {str(e)}")
        finally:
            if connection and not connection.is_closed:
                connection.close()
