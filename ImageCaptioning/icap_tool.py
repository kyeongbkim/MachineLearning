#!/usr/bin/env python

import pdb
import sys
import click
import collections
import image_captioning
import numpy as np
import pandas as pd

@click.group()
def cmd_group():
    pass

#------------------------------------------------------------------------------

def load_model(dataset, model):
    if model == 'cascade':
        import cascaded_encoder_decoder
        iCap = cascaded_encoder_decoder.ImageCaptioningWithCascadedEncoderDecoder(dataset)
    elif model == 'merge':
        import merged_encoder_decoder
        iCap = merged_encoder_decoder.ImageCaptioningWithMergedEncoderDecoder(dataset)
    elif model == 'attention':
        import attention
        iCap = attention.ImageCaptioningWithAttention(dataset)
    elif model == 'transformer':
        import transformer
        iCap = transformer.ImageCaptioningWithTransformer(dataset)
    else:
        assert False, 'Unknown model {}'.format(model)
    return iCap

def do_train(dataset, model, epochs):
    iCap = load_model(dataset, model)
    for epoch in epochs:
        iCap.train(epoch)
        iCap.save_weights(epoch)
    del iCap

def train(dataset, models, epochs):
    epochs = sorted([ int(e) for e in epochs.split(',')])
    for model in [ m for m in models.split(',') ]:
        print('Train {} model, Epochs: {}'.format(model, epochs))
        do_train(dataset, model, epochs)

@cmd_group.command('train')
@click.option('--dataset', '-d',
        default='Flickr30k',
        show_default=True,
        help='Dataset name')
@click.option('--models',  '-m',
        default='transformer,attention,merge,cascade',
        show_default=True,
        help='List of models')
@click.option('--epochs',  '-e',
        default='10,20,30,40,50',
        show_default=True,
        help='Save weights after given number of epochs')
def train_cmd(dataset, models, epochs):
    train(dataset, models, epochs)

#------------------------------------------------------------------------------

def do_predict(dataset, model, epochs, count):
    iCap = load_model(dataset, model)
    for epoch in [ int(e) for e in epochs.split(',')]:
        iCap.load_weights(epoch)

        num_images = len(iCap.caption_data.test_images)
        if count > 0 and count < num_images:
            num_images = count

        result = collections.OrderedDict()
        for i in range(num_images):
            image_id = iCap.caption_data.test_images[i]
            caption = iCap.predict(image_id)
            result[image_id] = caption
            print('[{}/{}] {}: {}'.format(i, num_images, image_id, caption))

        iCap.save_predictions(epoch, result)
    del iCap

def predict(dataset, models, epochs, count):
    for model in [ m for m in models.split(',') ]:
        print('Predict {} model, Epochs: {}'.format(model, epochs))
        do_predict(dataset, model, epochs, count)

@cmd_group.command('predict')
@click.option('--dataset', '-d',
        default='Flickr30k',
        show_default=True,
        help='Dataset name')
@click.option('--models',  '-m',
        default='transformer,attention,merge,cascade',
        show_default=True,
        help='List of models')
@click.option('--epochs',  '-e',
        default='10,20,30,40,50',
        show_default=True,
        help='Load pretrained weights after given number of epochs')
@click.option('--count',   '-n',
        default=0,
        show_default=True,
        help='Number of images to predict. (0 means all test images)')
def predict_cmd(dataset, models, epochs, count):
    predict(dataset, models, epochs, count)

#------------------------------------------------------------------------------

def get_eval_scores(references, prediction, rouge):
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import meteor_score

    scores = {}

    w1 = (1.0/1,   0.0,   0.0,   0.0)
    w2 = (1.0/2, 1.0/2,   0.0,   0.0)
    w3 = (1.0/3, 1.0/3, 1.0/3,   0.0)
    w4 = (1.0/4, 1.0/4, 1.0/4, 1.0/4)
    scores['b1'] = sentence_bleu(list(map(lambda ref: ref.split(), references)), prediction.split(), w1)
    scores['b2'] = sentence_bleu(list(map(lambda ref: ref.split(), references)), prediction.split(), w2)
    scores['b3'] = sentence_bleu(list(map(lambda ref: ref.split(), references)), prediction.split(), w3)
    scores['b4'] = sentence_bleu(list(map(lambda ref: ref.split(), references)), prediction.split(), w4)

    rouge_result = rouge.get_scores([prediction]*len(references), references)
    scores['r1'] = np.average([r['rouge-1']['f'] for r in rouge_result])
    scores['r2'] = np.average([r['rouge-2']['f'] for r in rouge_result])
    scores['rL'] = np.average([r['rouge-l']['f'] for r in rouge_result])

    scores['m'] = meteor_score(list(map(lambda ref: ref.split(), references)), prediction.split())
    return scores

def do_evaluate(dataset, model, epochs):
    from rouge import Rouge
    rouge = Rouge()

    iCap = load_model(dataset, model)

    for epoch in [ int(e) for e in epochs.split(',')]:
        predictions = iCap.load_predictions(epoch)
        df = pd.DataFrame(columns = ['image_id', 'b1', 'b2', 'b3', 'b4', 'r1', 'r2', 'rL', 'm' ])
        for image_id in iCap.caption_data.test_images:
            references = iCap.caption_data.get_captions(image_id)
            prediction = predictions[image_id]
            scores = get_eval_scores(references, prediction, rouge)
            scores['image_id'] = image_id
            df = df.append(scores, ignore_index=True)
        iCap.save_eval_scores(epoch, df)
    del iCap

def evaluate(dataset, models, epochs, consolidate):
    if consolidate:
        # Create intermediate data frame
        df = pd.DataFrame()
        for model in [ m for m in models.split(',') ]:
            iCap = load_model(dataset, model)
            for epoch in [ int(e) for e in epochs.split(',')]:
                scores = iCap.load_eval_scores(epoch)
                row = pd.concat([pd.Series({'model': model, 'epoch': epoch}), scores.mean(numeric_only=True)])
                df = pd.concat([df, pd.DataFrame(row).transpose()])

        # Build consolidated table
        metric_name = { 'b1': 'BLEU-1', 'b2': 'BLEU-2', 'b3': 'BLEU-3', 'b4': 'BLEU-4',
                        'r1': 'ROUGE-1', 'r2': 'ROUGE-2', 'rL': 'ROUGE-L',
                        'm': 'METEOR' }
        consolidated_df = pd.DataFrame(columns=['metric', 'epoch', 'cascade', 'merge', 'attention', 'transformer'])
        for metric in [ 'b1', 'b2', 'b3', 'b4', 'r1', 'r2', 'rL', 'm' ]:
            for epoch in [ int(e) for e in epochs.split(',')]:
                row = { 'metric': metric_name[metric],
                        'epoch': epoch,
                        'cascade': round(df[(df.model == 'cascade') & (df.epoch == epoch)][metric].values[0], 4),
                        'merge': round(df[(df.model == 'merge') & (df.epoch == epoch)][metric].values[0], 4),
                        'attention': round(df[(df.model == 'attention') & (df.epoch == epoch)][metric].values[0], 4),
                        'transformer': round(df[(df.model == 'transformer') & (df.epoch == epoch)][metric].values[0], 4) }
                consolidated_df = pd.concat([consolidated_df, pd.DataFrame(pd.Series(row)).transpose()], ignore_index=True)

        save_path = './workspace/{}-eval.csv'.format(dataset)
        print('Writing evaluation result to {}'.format(save_path))
        consolidated_df.to_csv(save_path, index=False)

    else:
        # Calculate evaluation scores
        for model in [ m for m in models.split(',') ]:
            print('Evaluate {} model, Epochs: {}'.format(model, epochs))
            do_evaluate(dataset, model, epochs)


@cmd_group.command('evaluate')
@click.option('--dataset', '-d',
        default='Flickr30k',
        show_default=True,
        help='Dataset name')
@click.option('--models',  '-m',
        default='transformer,attention,merge,cascade',
        show_default=True,
        help='List of models')
@click.option('--epochs',  '-e',
        default='10,20,30,40,50',
        show_default=True,
        help='Load pretrained weights after given number of epochs')
@click.option('--consolidate', '-c', is_flag=True,
        default=False,
        show_default=True,
        help='Create consolidated evaluation table')
def evaluate_cmd(dataset, models, epochs, consolidate):
    evaluate(dataset, models, epochs, consolidate)

#------------------------------------------------------------------------------

cli = click.CommandCollection(sources=[cmd_group])

if __name__ == '__main__':
    sys.exit(cli())
