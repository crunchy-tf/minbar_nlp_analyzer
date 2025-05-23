# nlp_analyzer_service/app/services/scheduler_service.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from app.config import settings # Use Pydantic settings
from app.main_processor import scheduled_nlp_job # Import the job

scheduler = AsyncIOScheduler(timezone="UTC")
_scheduler_started = False

async def start_scheduler():
    global _scheduler_started
    if scheduler.running or _scheduler_started:
        logger.info("NLP Analyzer APScheduler already running or start initiated.")
        return
    try:
        scheduler.add_job(
            scheduled_nlp_job,
            trigger=IntervalTrigger(minutes=settings.SCHEDULER_INTERVAL_MINUTES),
            id="nlp_processing_job",
            name="Poll and Process Documents for NLP Analysis",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=300 
        )
        scheduler.start()
        _scheduler_started = True
        logger.success(f"NLP Analyzer APScheduler started. Job scheduled every {settings.SCHEDULER_INTERVAL_MINUTES} minutes.")
    except Exception as e:
        logger.error(f"Failed to start NLP Analyzer APScheduler: {e}", exc_info=True)
        _scheduler_started = False


async def stop_scheduler():
    global _scheduler_started
    if scheduler.running:
        logger.info("Stopping NLP Analyzer APScheduler...")
        scheduler.shutdown(wait=False)
        _scheduler_started = False
        logger.success("NLP Analyzer APScheduler stopped.")