import logging
import json
import os
from datetime import datetime, timedelta, timezone

from exp.agent.log import LogAgent

def setup_logger():
    """日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def run_log_analysis_in_chunks():
    logger = logging.getLogger(__name__)
    
    DATASET_ROOT = './phaseone'
    TOTAL_START_TIME_STR = "2025-06-06 00:00:00"
    TOTAL_END_TIME_STR = "2025-06-06 23:59:59"
    CHUNK_SIZE_HOURS = 1
    OUTPUT_FILENAME = "./log_results/phaseone/log_analysis_report.jsonl"

    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)

    total_start_time_utc = datetime.strptime(TOTAL_START_TIME_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    total_end_time_utc = datetime.strptime(TOTAL_END_TIME_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    log_agent = LogAgent(DATASET_ROOT)
    all_results = []

    current_time_utc = total_start_time_utc
    while current_time_utc < total_end_time_utc:
        chunk_start_utc = current_time_utc
        chunk_end_utc = current_time_utc + timedelta(hours=CHUNK_SIZE_HOURS)
        if chunk_end_utc > total_end_time_utc:
            chunk_end_utc = total_end_time_utc

        time_compensation = timedelta(hours=8)
        compensated_start = chunk_start_utc - time_compensation
        compensated_end = chunk_end_utc - time_compensation

        print("\n" + "="*20 + f" Processing chunk: {chunk_start_utc} to {chunk_end_utc} " + "="*20)

        try:
            results_chunk = log_agent.score(compensated_start, compensated_end, max_workers=12)

            if results_chunk and len(results_chunk) > 0:
                all_results.extend(results_chunk)
                logger.info(f"Chunk processed successfully, found {len(results_chunk)} results.")
            else:
                logger.info(f"No anomalies found in this chunk.")

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_start_utc}: {e}", exc_info=True)
        
        current_time_utc += timedelta(hours=CHUNK_SIZE_HOURS)
    with open(OUTPUT_FILENAME, "w") as f_out:
        for result in all_results:
            f_out.write(json.dumps(result) + '\n')
    logger.info(f"Log analysis completed. Total results: {len(all_results)}. Saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    setup_logger()
    run_log_analysis_in_chunks()
