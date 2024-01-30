import os
import json
import time
import argparse
 
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
 
MAX_SEQ_LEN = 512
 
def get_context_data(tokenizer):
    start_path = './abstract_data/'
    multiple_part_count = 0
    for json_file in os.listdir(start_path):
        # subject = json_file.split('-')[0]
        with open(os.path.join(start_path, json_file), encoding='utf-8') as f:
            meta_datas = json.load(f)
            for meta_data in meta_datas:
                # yield subject, meta_data
                encode_token = tokenizer(meta_data['摘要'])['input_ids']
                if (total_len:=len(encode_token)) >= MAX_SEQ_LEN:
                    # print(total_len, meta_data)
                    multiple_part_count += 1
                    print('New multiple corpus:\t', multiple_part_count)
                slice_part = [encode_token[i:i + MAX_SEQ_LEN] for i in range(0, len(encode_token), MAX_SEQ_LEN)]
                for ind, slice in enumerate(slice_part):
                    curr_abstract = tokenizer.decode(slice)
                    meta_data['词嵌入语料'] = curr_abstract
                    meta_data['部分'] = ind + 1
                    yield meta_data['词嵌入语料']
 
def main(batch_size: int = 512, batch_time: float = 0.1, model_name:str='thenlper/gte-base'):
    start_t, last_batch_t= time.time(), time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentenceTransformer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('Model load completed, start to do inference...')
    print(f'MODEL NAME: {model_name}, BATCH_SIZE: {batch_size}')
    curr_batch = []
    ind, batch_count = 0, 0
    for data in get_context_data(tokenizer=tokenizer):
        ind += 1
        curr_batch.append(data)
        if not ind%batch_size:
            batch_count += 1
            model.encode(curr_batch)
            print('Curr Batch: %s\tCurr Batch time: %.4fs\tTotal Time Use: %.4fs'%(batch_count, time.time() - last_batch_t, time.time() - start_t))
            last_batch_t = time.time()
            curr_batch = []
    if curr_batch:
        model.encode(curr_batch)
        batch_count += 1
    db_write_time = batch_time * batch_count
    print('Finish all %s corpus with in %.4fs, multiple corpus count %s.'%(ind, time.time() - start_t + db_write_time, batch_count))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The parser for the main function.')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='thenlper/gte-base')
    args = parser.parse_args()
    main(batch_size=args.batch_size, model_name=args.model_name)