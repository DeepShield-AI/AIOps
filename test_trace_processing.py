import logging
import json
from datetime import datetime, timedelta, timezone

from exp.agent.trace import TraceAgent

def setup_logger():
    """Setup logger"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def run_trace_analysis_in_chunks():
    """
    Run trace data analysis in chunks (hourly) and write final results to a file.
    """
    logger = logging.getLogger(__name__)
    
    DATASET_ROOT = './phaseone'
    TOTAL_START_TIME_STR = "2025-06-06 00:00:00"
    TOTAL_END_TIME_STR = "2025-06-06 05:59:59"
    CHUNK_SIZE_HOURS = 1
    OUTPUT_FILENAME = "trace_analysis_report.jsonl"

    logger.info(f"Starting trace analysis (chunked mode)...")
    logger.info(f"Dataset root: {DATASET_ROOT}")

    total_start_time_utc = datetime.strptime(TOTAL_START_TIME_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    total_end_time_utc = datetime.strptime(TOTAL_END_TIME_STR, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    
    logger.info(f"Target analysis time range: {total_start_time_utc} to {total_end_time_utc}")

    trace_agent = TraceAgent(root_path=DATASET_ROOT)
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

        print("\n" + "="*20 + f" Processing time chunk: {chunk_start_utc} to {chunk_end_utc} " + "="*20)

        try:
            results_chunk = trace_agent.score(compensated_start, compensated_end, max_workers=12)
            if len(results_chunk) != 0:
            # if results_chunk and "No spans found" not in results_chunk[0] and "No anomalous traces found" not in results_chunk[0]:
                all_results.extend(results_chunk)
                logger.info(f"Chunk processed, found {len(results_chunk)} results.")
            else:
                logger.info(f"No anomalies found in this chunk.")

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_start_utc}: {e}", exc_info=True)
        
        current_time_utc += timedelta(hours=CHUNK_SIZE_HOURS)


    logger.info(f"All chunks processed, writing results to file: {OUTPUT_FILENAME}")
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            # f.write("="*25 + " Trace Analysis Report " + "="*25 + "\n\n")
            for result in all_results:
                f.write(json.dumps(result) + '\n')
            # f.write("\n" + "="*65 + "\n")
        
        print(f"\nAnalysis complete. Results written to {OUTPUT_FILENAME}.")

    except IOError as e:
        logger.error(f"Failed to write file {OUTPUT_FILENAME}: {e}")
        print(f"\nError: Unable to write results to file {OUTPUT_FILENAME}. Please check permissions.")


if __name__ == "__main__":
    setup_logger()
    run_trace_analysis_in_chunks()
