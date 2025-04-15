import json
import time
import math
import asyncio
import os
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from tqdm.asyncio import tqdm_asyncio

TOTAL_PRODUCTS_TO_GENERATE = 10
BATCH_SIZE = 2
MAX_CONCURRENT_REQUESTS = 3
OUTPUT_FILENAME = f"fmcg_data_{TOTAL_PRODUCTS_TO_GENERATE}.json"
LOG_FILENAME = "log.txt"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 5

print("--- Initializing LLM ---")
print("Ensure Ollama server is running and using GPU if available.")
llm = ChatOllama(
    model="llama3.2",
    temperature=0.2,
    num_gpu=1
)
print("LLM Initialized.")
print("-" * 30)


def create_prompt(num_products):
    return f"""
Generate {num_products} unique and realistic FMCG (Fast-Moving Consumer Goods) product entries as a JSON list.

Each item in the list must follow this exact format:
{{
  "id": "string (unique alphanumeric product ID, e.g., FMCG-XXXX)",
  "title": "string (brief, market-ready product title)",
  "description": "string (a richly detailed product description including its use-case, key features, and where applicable: nutritional information for food/beverage or ingredients for personal care and cleaning products. Min 50 words.)",
  "category": "string (one of: Beverages, Snacks, Personal Care, Cleaning, Household, Dairy, Bakery, Frozen Foods, Canned Goods, Baby Care, Pet Care)",
  "brand": "string (realistic-sounding or fictional brand name)",
  "price": float (reasonable price in USD, e.g. 3.99),
  "tags": ["string", ...] (array of 3-7 relevant keywords such as 'organic', 'sugar-free', 'vegan', 'eco-friendly', 'gluten-free', 'family size', 'quick-clean', etc.)
}}

Guidelines:
- Ensure IDs are unique within the batch.
- For **food or beverage products**, include a short nutrition facts section in the description (e.g., calories, fats, proteins, sugars per serving).
- For **personal care, household, or cleaning products**, include key **ingredients or active compounds** (e.g., aloe vera, sodium lauryl sulfate, bleach-free).
- Ensure diversity across categories, brands, and price points.
- Descriptions should be informative, persuasive, and realistic (min 50 words).
- Make sure each product is **unique** within this batch, and the tone stays consistent.

IMPORTANT: Output ONLY the raw JSON list starting with '[' and ending with ']'. Do NOT include ```json ``` markers, explanations, or any other text outside the JSON list itself. The output must be a single JSON list containing exactly {num_products} product objects.
"""

def extract_json_list(text):
    match = re.search(r"\[.*\]", text.strip(), re.DOTALL)
    if match:
        potential_json = match.group(0)
        if potential_json.startswith('[') and potential_json.endswith(']') and '{' in potential_json and '}' in potential_json:
             return potential_json

    cleaned_text = text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:-3].strip()
    elif cleaned_text.startswith("```"):
         cleaned_text = cleaned_text[3:-3].strip()

    if cleaned_text.startswith('[') and cleaned_text.endswith(']') and '{' in cleaned_text and '}' in cleaned_text:
        return cleaned_text

    print(f"\nWarning: Could not reliably extract JSON list structure. Raw text snippet: {text[:200]}...")
    return text.strip()


async def generate_batch(batch_num, num_products_in_batch, semaphore):
    async with semaphore:
        batch_prompt = create_prompt(num_products_in_batch)
        for attempt in range(MAX_RETRIES + 1):
            response_content = ""
            try:
                response = await llm.ainvoke([HumanMessage(content=batch_prompt)])
                response_content = response.content
                json_string_to_parse = extract_json_list(response_content)
                batch_data = json.loads(json_string_to_parse)

                if isinstance(batch_data, list) and len(batch_data) == num_products_in_batch:
                    return batch_data
                else:
                    print(f"\nWarning in Batch {batch_num}, Attempt {attempt+1}: Validation failed. Expected {num_products_in_batch} items, got {len(batch_data) if isinstance(batch_data, list) else 'non-list type'}.")
                    raise ValueError("Validation failed: Incorrect item count or type.")

            except json.JSONDecodeError as e:
                print(f"\nError in Batch {batch_num}, Attempt {attempt+1}: JSONDecodeError - {e}")
                print(f"LLM Response Snippet (Batch {batch_num}):\n{response_content[:200]}...")
            except ValueError as e:
                 print(f"\nError in Batch {batch_num}, Attempt {attempt+1}: {e}")
            except Exception as e:
                print(f"\nError in Batch {batch_num}, Attempt {attempt+1}: Unexpected {type(e).__name__} - {e}")

            if attempt < MAX_RETRIES:
                print(f"Batch {batch_num}: Retrying after {RETRY_DELAY_SECONDS}s delay...")
                await asyncio.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"Batch {batch_num}: Failed after {MAX_RETRIES + 1} attempts.")
                return []

        return []


async def run_generation():
    all_products = []
    num_batches = math.ceil(TOTAL_PRODUCTS_TO_GENERATE / BATCH_SIZE)

    print(f"Starting generation of {TOTAL_PRODUCTS_TO_GENERATE} products.")
    print(f"Config: Batch Size={BATCH_SIZE}, Max Concurrent={MAX_CONCURRENT_REQUESTS}, Retries={MAX_RETRIES}")
    print(f"Total Batches: {num_batches}")
    print("-" * 30)

    start_time = time.time()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    products_to_request = TOTAL_PRODUCTS_TO_GENERATE

    for i in range(num_batches):
        current_batch_size = min(BATCH_SIZE, products_to_request)
        if current_batch_size <= 0:
            break
        task = generate_batch(i + 1, current_batch_size, semaphore)
        tasks.append(task)
        products_to_request -= current_batch_size

    results = await tqdm_asyncio.gather(*tasks, desc="Generating Batches", unit="batch")

    end_time = time.time()
    duration = end_time - start_time

    successful_batches = 0
    failed_batches = 0
    for batch_result in results:
        if isinstance(batch_result, list) and batch_result:
            all_products.extend(batch_result)
            successful_batches += 1
        else:
            failed_batches += 1

    products_generated_count = len(all_products)
    print(f"\n--- Generation Complete ---")
    print(f"Target: {TOTAL_PRODUCTS_TO_GENERATE}, Successfully Generated: {products_generated_count}")
    print(f"Successful Batches: {successful_batches}/{num_batches}, Failed Batches: {failed_batches}")
    print(f"Total Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    if all_products:
        print(f"Saving {products_generated_count} products to {OUTPUT_FILENAME}...")
        try:
            with open(OUTPUT_FILENAME, "w") as f:
                json.dump(all_products, f, indent=2)
            print(f"Successfully saved to {OUTPUT_FILENAME}")
            file_size = os.path.getsize(OUTPUT_FILENAME)
            print(f"Output file size: {file_size / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        print("No products were generated successfully.")

    log_message = (
        f"{products_generated_count}/{TOTAL_PRODUCTS_TO_GENERATE} products | "
        f"Batch Size: {BATCH_SIZE} | Concurrency: {MAX_CONCURRENT_REQUESTS} | "
        f"Time: {duration:.2f}s ({duration/60:.2f}m) | "
        f"Success Rate: {successful_batches}/{num_batches} batches\n"
    )
    try:
        with open(LOG_FILENAME, "a") as log_file:
            log_file.write(log_message)
    except Exception as e:
        print(f"Error writing to log file: {e}")

if __name__ == "__main__":
    try:
        from tqdm.asyncio import tqdm_asyncio
    except ImportError:
        print("Error: 'tqdm' package not found. Please install it: pip install tqdm")
        exit(1)

    asyncio.run(run_generation())
