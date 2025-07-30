import json

def split_text(text, max_len=512):
    # 按段落分割
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 1 <= max_len:
            buf += (para if not buf else '\n' + para)
        else:
            if buf:
                chunks.append(buf)
            if len(para) > max_len:
                # 句号分割长段
                sents = para.split('。')
                sent_buf = ""
                for sent in sents:
                    if not sent:
                        continue
                    sent += '。'
                    if len(sent_buf) + len(sent) <= max_len:
                        sent_buf += sent
                    else:
                        if sent_buf:
                            chunks.append(sent_buf)
                        sent_buf = sent
                if sent_buf:
                    chunks.append(sent_buf)
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)
    return chunks

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

samples = split_text(text, max_len=512)

with open('data.jsonl', 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps({"text": sample}, ensure_ascii=False) + '\n')

print(f"已生成 {len(samples)} 条样本到 data.jsonl")