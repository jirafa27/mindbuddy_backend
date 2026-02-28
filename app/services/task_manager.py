from app.infrastructure.workers.celery_service import CeleryService
from app.infrastructure.message_broker.rabbitmq import RabbitMQService
import asyncio
import logging

logger = logging.getLogger(__name__)

class TaskManager:
	def __init__(self, celery_service: CeleryService, rabbitmq_service: RabbitMQService):
		self.celery_service = celery_service
		self.rabbitmq_service = rabbitmq_service

	def send_file_embedding_task(self, file_id, text, namespace_id, filename, user_file_id):
		asyncio.run(self.celery_service.send_file_embedding_task(file_id, text, namespace_id, filename, user_file_id))