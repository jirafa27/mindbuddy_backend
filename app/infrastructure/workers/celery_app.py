from celery import Celery

from app.core.config import settings

celery_app = Celery(
		"mindbuddy",
		broker=settings.RABBITMQ_URL,
		backend=settings.REDIS_URL,
		include=[
			"app.infrastructure.workers.file_processing",
			"app.infrastructure.workers.summary_processing",
		],
	)

celery_app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="UTC",
            enable_utc=True,
            broker_connection_retry_on_startup=True,
        )

