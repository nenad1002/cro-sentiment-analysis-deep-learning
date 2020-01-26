import io
import json

def generate_embeddings(model, tokenizer):
    '''
    Generates embeddings vector with metadata.
    Use to preview: http://projector.tensorflow.org/
    '''
    e = model.layers[0]
    weights = e.get_weights()[0]

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')

    num_of_words = tokenizer.get_config()['num_words']
    index_to_word = json.loads(tokenizer.get_config()['index_word'])

    for num in range(num_of_words - 1):
        # Skip 0, it's padding.
        vec = weights[num + 1]
        word = index_to_word[str(num + 1)]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")

    out_v.close()
    out_m.close()