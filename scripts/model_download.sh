mkdir hf_models
cd hf_models/
hdfs dfs get hdfs://haruna/home/byte_data_seed/ssd_hldy/evals_pipeline/open_source_models/llama3/Meta-Llama-3-8B
hdfs dfs ls hdfs://haruna/home/byte_data_seed/ssd_hldy/evals_pipeline/open_source_models/deepseek-ai
hdfs dfs get hdfs://haruna/home/byte_data_seed/ssd_hldy/evals_pipeline/open_source_models/deepseek-ai/deepseek-llm-7b-base
hdfs dfs get hdfs://haruna/home/byte_data_seed/ssd_hldy/evals_pipeline/open_source_models/deepseek-ai/DeepSeek-V2-Lite
