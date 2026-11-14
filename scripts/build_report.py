#!/usr/bin/env python3
"""Build a concise academic PDF from the frozen philosophy experiment.

Measurements come exclusively from results.json. The optional repository URL
must refer to the verified published repository, supplied by the caller.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 36 * mm
INK, MUTED = colors.black, colors.black
RULE, LIGHT = colors.black, colors.HexColor('#EAEAEA')
BLUE, ORANGE = colors.black, colors.black


def register_fonts():
    options = [
        (Path('/System/Library/Fonts/Supplemental'), 'Times New Roman.ttf', 'Times New Roman Bold.ttf', 'Times New Roman Italic.ttf'),
        (Path('/usr/share/fonts/truetype/dejavu'), 'DejaVuSerif.ttf', 'DejaVuSerif-Bold.ttf', 'DejaVuSerif-Italic.ttf'),
    ]
    for directory, regular, bold, italic in options:
        if (directory / regular).exists():
            for name, filename in [('Paper', regular), ('PaperBold', bold), ('PaperItalic', italic)]:
                pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
            pdfmetrics.registerFontFamily('Paper', normal='Paper', bold='PaperBold', italic='PaperItalic', boldItalic='PaperBold')
            return
    raise RuntimeError('Install Times New Roman or DejaVu Serif to build the report.')


def make_styles():
    styles = getSampleStyleSheet()
    definitions = {
        'Title': dict(fontName='Paper', fontSize=18, leading=22, spaceAfter=4, textColor=INK, alignment=TA_CENTER),
        'Course': dict(fontName='Paper', fontSize=11.5, leading=14, spaceAfter=11, textColor=INK, alignment=TA_CENTER),
        'Identity': dict(fontName='Paper', fontSize=11, leading=14, textColor=INK, alignment=TA_CENTER),
        'Date': dict(fontName='Paper', fontSize=11, leading=14, spaceBefore=10, spaceAfter=17, textColor=INK, alignment=TA_CENTER),
        'AbstractHeading': dict(fontName='PaperBold', fontSize=11, leading=14, spaceAfter=7, textColor=INK, alignment=TA_CENTER),
        'Body': dict(fontName='Paper', fontSize=10.5, leading=13.9, spaceAfter=7, textColor=INK, alignment=TA_JUSTIFY),
        'Heading': dict(fontName='PaperBold', fontSize=12, leading=15, spaceBefore=10, spaceAfter=6, textColor=INK),
        'Subheading': dict(fontName='PaperBold', fontSize=10.6, leading=14, spaceBefore=7, spaceAfter=4, textColor=INK),
        'Small': dict(fontName='Paper', fontSize=9, leading=11.8, spaceAfter=6, textColor=MUTED),
        'Abstract': dict(fontName='Paper', fontSize=10, leading=13, leftIndent=12*mm, rightIndent=12*mm, spaceAfter=7, textColor=INK, alignment=TA_JUSTIFY),
        'Equation': dict(fontName='Paper', fontSize=10.3, leading=15.2, spaceBefore=3, spaceAfter=6, leftIndent=13, textColor=INK),
        'Table': dict(fontName='Paper', fontSize=8.9, leading=11.3, textColor=INK),
        'TableHead': dict(fontName='PaperBold', fontSize=8.8, leading=11.3, textColor=INK),
        'Code': dict(fontName='Courier', fontSize=8, leading=11, leftIndent=8, spaceBefore=3, spaceAfter=7, textColor=INK),
    }
    for name, kwargs in definitions.items():
        styles.add(ParagraphStyle('Paper'+name, **kwargs))
    return styles


def academic_table(headers, rows, widths, styles):
    values = [[Paragraph(escape(str(item)), styles['PaperTableHead']) for item in headers]]
    values += [[Paragraph(str(item), styles['PaperTable']) for item in row] for row in rows]
    table = Table(values, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), .7, INK),
        ('LINEBELOW', (0, 0), (-1, 0), .5, RULE),
        ('LINEBELOW', (0, -1), (-1, -1), .7, INK),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    return table


class LearningFigure(Flowable):
    """One compact vector figure with HMM and RNN learning-curve panels."""
    def __init__(self, hmm, rnn, width=WIDTH, height=185):
        super().__init__()
        self.width, self.height = width, height
        self.models = [hmm, rnn]

    def draw(self):
        canvas, panel = self.canv, self.width / 2
        for index, model in enumerate(self.models):
            left = index * panel
            x0, y0 = left + 36, 29
            width, height = panel - 49, self.height - 63
            history = model['history']
            xs = [row['epoch'] for row in history]
            ys = [row[key] for row in history for key in ('train_nll', 'validation_nll')]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            padding = max(.03, (ymax-ymin)*.06)
            ymin, ymax = ymin-padding, ymax+padding
            def x(value): return x0+(value-xmin)/max(1,xmax-xmin)*width
            def y(value): return y0+(value-ymin)/max(.01,ymax-ymin)*height
            canvas.setFont('PaperBold', 9.4)
            canvas.setFillColor(INK)
            canvas.drawCentredString(left+panel/2, self.height-10, model['display_name'])
            canvas.setFont('Paper', 8)
            for tick in range(4):
                value = ymin+tick*(ymax-ymin)/3
                canvas.setStrokeColor(LIGHT)
                canvas.line(x0,y(value),x0+width,y(value))
                canvas.setFillColor(MUTED)
                canvas.drawRightString(x0-5,y(value)-2.5,f'{value:.2f}')
            for value in sorted(set([xmin,int((xmin+xmax)/2),xmax])):
                canvas.drawCentredString(x(value),y0-11,str(value))
            canvas.setStrokeColor(RULE)
            canvas.line(x0,y0,x0,y0+height)
            canvas.line(x0,y0,x0+width,y0)
            for key,color in [('train_nll',BLUE),('validation_nll',ORANGE)]:
                canvas.setStrokeColor(color)
                canvas.setLineWidth(1.3)
                canvas.setDash([3,2] if key=='validation_nll' else [])
                for first,second in zip(history[:-1],history[1:]):
                    canvas.line(x(first['epoch']),y(first[key]),x(second['epoch']),y(second[key]))
            canvas.setDash([])
            retained = next((row for row in history if row['epoch']==model['best_epoch']),None)
            if retained:
                canvas.setFillColor(ORANGE)
                canvas.circle(x(retained['epoch']),y(retained['validation_nll']),2.4,fill=1,stroke=0)
            canvas.setFillColor(MUTED)
            canvas.setFont('Paper',8)
            canvas.drawCentredString(x0+width/2,3,'EM update' if model['family']=='HMM' else 'Training epoch')
            canvas.saveState()
            canvas.translate(left+8,y0+height/2)
            canvas.rotate(90)
            canvas.drawCentredString(0,0,'NLL per token')
            canvas.restoreState()
        legend_x = self.width/2-55
        for label,color in [('Train',BLUE),('Validation',ORANGE)]:
            canvas.setStrokeColor(color)
            canvas.setDash([3,2] if label=='Validation' else [])
            canvas.line(legend_x,self.height-25,legend_x+11,self.height-25)
            canvas.setDash([])
            canvas.setFillColor(INK)
            canvas.setFont('Paper',8)
            canvas.drawString(legend_x+15,self.height-28,label)
            legend_x += 58


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Paper',9)
    canvas.setFillColor(INK)
    canvas.drawCentredString(PAGE_W/2,8.8*mm,str(doc.page))
    canvas.restoreState()


def display_name(row):
    parameters=row['parameters']
    if row['family']=='HMM':
        return f"HMM K={parameters['n_states']}, s={parameters['seed']}"
    if row['family']=='RNN':
        return f"RNN H={parameters['hidden_size']}, s={parameters['seed']}"
    return row['name'].capitalize()


def read_experiment(path):
    results=json.loads(path.read_text())
    title=results['corpus']['source']['title']
    if 'varieties' not in title.lower() or not any(row['family']=='RNN' for row in results['models']):
        raise ValueError('This report requires the upgraded William James experiment including RNN results.')
    for row in results['models']:
        row['display_name']=display_name(row)
    return results


def training_description(results):
    models,grid=results['models'],results['config']['baseline_grid']
    alphas=', '.join(map(str,grid['alpha']))
    weights=', '.join(map(str,grid['interpolation']))
    descriptions=[f'Each n-gram order searches a in {{{alphas}}} and λ in {{{weights}}}; additive smoothing applies only to its unigram base.']
    for row in models:
        if row['family'] in ('bigram','trigram'):
            parameters=row['parameters']
            descriptions.append(f"The retained {row['family']} uses a={parameters['alpha']}, λ={parameters['interpolation']}.")
    hmm=next(row for row in models if row['family']=='HMM')['parameters']
    rnn=next(row for row in models if row['family']=='RNN')['parameters']
    hidden=sorted({row['parameters']['hidden_size'] for row in models if row['family']=='RNN'})
    descriptions.append(f"HMMs use at most {hmm['n_iter']} EM updates, with validation patience {hmm['patience']}.")
    descriptions.append(f"RNNs use {rnn['embedding_dim']}-dimensional embeddings, hidden sizes {', '.join(map(str,hidden))}, "
                        f"and at most {rnn['n_epochs']} epochs with patience {rnn['patience']}.")
    descriptions.append('All early stopping and checkpoint selection use validation NLL. Patience requires improvement above 0.0001; the absolute minimum NLL still selects the retained checkpoint.')
    return ' '.join(descriptions)


def checkpoint_description(hmm,rnn):
    clauses=[]
    for model in (hmm,rnn):
        last=max(row['epoch'] for row in model['history'])
        unit='EM updates' if model['family']=='HMM' else 'epochs'
        clauses.append(f"{escape(model['display_name'])} retains checkpoint {model['best_epoch']} from {last} completed {unit}")
    return '; '.join(clauses)+'. A run that reaches its maximum budget is not evidence of convergence. The curves expose training/validation divergence and the effect of checkpoint selection.'


def build_report(results,output,repository_url='',author='Avi Berezovski',student_id='',report_date='November 2026'):
    if student_id and output.resolve()==(ROOT/'output/pdf/project_report.pdf').resolve():
        raise ValueError('Choose a separate --output path for a report containing a student ID.')
    register_fonts()
    styles,story=make_styles(),[]
    models,corpus,config=results['models'],results['corpus'],results['config']
    best=min(models,key=lambda row:row['validation']['nll'])
    best_hmm=min((row for row in models if row['family']=='HMM'),key=lambda row:row['validation']['nll'])
    best_rnn=min((row for row in models if row['family']=='RNN'),key=lambda row:row['validation']['nll'])
    best_count=min((row for row in models if row['family'] not in ('HMM','RNN')),key=lambda row:row['validation']['nll'])
    unigram=next(row for row in models if row['family']=='unigram')
    score=best['test']
    def p(text,kind='Body'): story.append(Paragraph(text,styles['Paper'+kind]))
    def heading(text): p(text,'Heading')
    def subheading(text): p(text,'Subheading')
    def eq(text): p(text,'Equation')
    def new_page(): story.append(PageBreak())

    # Page 1: academic title, abstract, motivation, and dataset; no cover page.
    p('Next-Word Prediction in Philosophy of Religion:<br/>N-gram, Hidden Markov and Recurrent Neural Models','Title')
    p('Bayesian Inference Course Project','Course')
    if author:
        p(escape(author),'Identity')
    if student_id:
        p('ID: '+escape(student_id),'Identity')
    p(escape(report_date),'Date')
    p('Abstract','AbstractHeading')
    p('This project implements next-word completion for a domain-focused English corpus drawn from William James\'s <i>The Varieties of Religious Experience</i>. '
      'Unigram, interpolated n-gram, categorical hidden Markov, and Elman recurrent neural models are compared using a common training-only vocabulary and held-out text. '
      'The experiment varies HMM state count and RNN hidden size, selects parameters and checkpoints on validation negative log-likelihood, and reports test perplexity and lexical Top-k accuracy. '
      f"The validation-selected model, {escape(best['display_name'])}, obtains test perplexity {score['perplexity']:.2f} and Top-3 accuracy {100*score['top3']:.2f}%. "
      'The comparison separates probability estimation from useful word suggestions and documents out-of-vocabulary effects, training budgets, and limitations of a single-book study.','Abstract')
    heading('1. Introduction')
    p('An autocomplete system estimates P(w<sub>t+1</sub> | w<sub>1:t</sub>) and ranks likely continuations of the text typed so far. '
      'This project studies the task in writing about religion and religious experience, where recurring conceptual vocabulary coexists with narrative quotations and philosophical argument. '
      'The corpus supplies a coherent subject area; the models are language predictors and do not evaluate the truth or meaning of religious claims.')
    p('The central question is whether latent sequence models improve on inexpensive local word counts under the same data and evaluation conditions. '
      'HMMs compress a prefix into a distribution over discrete states, while an Elman RNN learns a continuous recurrent representation. '
      'Varying capacity and recording validation learning curves makes the comparison more informative than testing one configuration per family.')
    heading('2. Corpus and preprocessing')
    p('The data are the twenty main lectures of William James\'s <i>The Varieties of Religious Experience: A Study in Human Nature</i>, from Project Gutenberg eBook 621 [1]. '
      'The raw text and its source URL, retrieval date, and SHA-256 digest are retained. Preface, contents, postscript, index, and endnotes are excluded; lecture headings and attached numeric footnote markers are removed. '
      'Paragraphs are split in source order: 80% training, 10% validation, and 10% test. Sentence segmentation follows the split, so no source paragraph spans two subsets.')
    rows=[]
    for split in ('train','validation','test'):
        values=corpus[split]
        rows.append([split.capitalize(),f"{values['paragraphs']:,}",f"{values['sentences']:,}",f"{values['lexical_tokens']:,}",f"{values['tokens_including_eos']:,}",f"{100*values['oov_rate']:.2f}%"])
    story.append(academic_table(['Split','Paragraphs','Sentences','Lexical tokens','With EOS','OOV'],rows,[65,74,75,106,91,WIDTH-411],styles))
    p(f"Table 1. Dataset split. Lowercase tokens retain numbers, internal apostrophes/hyphens, and decimals. The {corpus['vocabulary_size']:,}-symbol vocabulary uses training data only and includes &lt;unk&gt; and &lt;eos&gt;. "
      'Rare or unseen words map to UNK; each independent sentence ends with EOS. Sentence and abbreviation handling are heuristic.','Small')

    # Page 2: exact implemented algorithms and fair scoring.
    new_page()
    heading('3. Implemented models')
    subheading('3.1. Unigram and interpolated n-gram baselines')
    p('The unigram uses add-a smoothing over N training tokens and a V-symbol vocabulary. For a seen history h, a higher-order model interpolates the empirical conditional distribution with the shorter-history model; unseen histories fall back completely. '
      'The same interpolation weight is used at both trigram levels. Internal BOS markers provide the initial context and are never predicted.')
    eq('P<sub>1</sub>(w) = [C(w) + a] / [N + aV],<br/>P<sub>n</sub>(w | h) = λ P<sub>ML</sub>(w | h) + (1 - λ) P<sub>n-1</sub>(w | suffix(h)).')
    subheading('3.2. Hidden Markov model and Baum-Welch')
    p('A categorical HMM has K latent states, initial probabilities π, transitions A<sub>ij</sub> = P(z<sub>t+1</sub>=j | z<sub>t</sub>=i), and emissions B<sub>iw</sub> = P(w<sub>t</sub>=w | z<sub>t</sub>=i). '
      'For the filtered row vector q<sub>t</sub> = P(z<sub>t</sub> | w<sub>1:t</sub>), the next-word distribution is q<sub>t</sub>AB; at empty context it is πB. '
      'Forward vectors are normalized at each step, and summed log normalization factors give sequence log-likelihood.')
    p('Baum-Welch alternates a forward-backward E-step, calculating posterior occupancies γ and transitions ξ, with an expected-count M-step [2,3]:')
    eq('γ<sub>t</sub>(i) = α<sub>t</sub>(i) β<sub>t</sub>(i) / Σ<sub>j</sub> α<sub>t</sub>(j) β<sub>t</sub>(j),<br/>'
       'ξ<sub>t</sub>(i,j) proportional to α<sub>t</sub>(i) A<sub>ij</sub> B<sub>j,w(t+1)</sub> β<sub>t+1</sub>(j),<br/>'
       'π<sub>i</sub> ← mean<sub>s</sub> γ<sub>s,1</sub>(i); A<sub>ij</sub> ← Σ ξ(i,j) / Σ γ(i);<br/>'
       'B<sub>iw</sub> ← Σ γ(i) 1[w<sub>t</sub>=w] / Σ γ(i).')
    p('Counts are summed over independent training sentences; transition sums exclude final positions. A pseudocount of 0.01 precedes row normalization. '
      'This regularized point estimation does not sample a Bayesian parameter posterior; raw-likelihood monotonicity applies to unsmoothed EM, not necessarily smoothed updates. '
      'Hidden states are not labelled as grammatical or conceptual categories.','Small')
    subheading('3.3. Elman recurrent neural language model')
    p('An input token maps to a learned embedding e<sub>t</sub>. The recurrence updates a continuous hidden state; a softmax layer estimates the next token [2,4]:')
    eq('h<sub>t</sub> = tanh(W e<sub>t</sub> + U h<sub>t-1</sub> + b),<br/>P(w<sub>t+1</sub> | w<sub>1:t</sub>) = softmax(V h<sub>t</sub> + c).')
    p('Full-sentence backpropagation through time minimizes cross-entropy. A BOS input predicts the first word, subsequent words predict their successors, and the final word predicts EOS. '
      'States reset per sentence. Adam uses learning rate 0.003, batch size 64, and gradient-norm clipping at 5.0; there is no dropout or truncated context. '
      'Padding positions are masked. A fixed seed and deterministic CPU operations aid reproducibility.')
    heading('4. Evaluation protocol')
    p('Every target is predicted from preceding true tokens within its sentence. Mean negative natural log-likelihood (NLL) includes first words, UNK, and EOS; perplexity is exp(NLL). '
      'Lexical Top-1/3/5 accuracy excludes EOS targets, removes UNK/EOS from suggestions, and counts unknown lexical targets as misses. '
      'Validation selects hyperparameters, checkpoints, and the deployed model. Test scoring occurs only after these choices are frozen.')

    # Page 3: eleven measured rows, budgets, and numerical interpretation.
    new_page()
    heading('5. Quantitative results')
    p('Table 2 reports the validation-selected baseline per n-gram order and every prespecified HMM/RNN configuration. Lower perplexity and higher lexical accuracy are preferable. '
      'Perplexities are comparable within this vocabulary, not directly with scores from a different tokenizer or vocabulary.')
    rows=[]
    for row in models:
        metrics=row['test']
        label=escape(row['display_name'])
        if row['name']==best['name']: label=f'<b>{label}</b>'
        rows.append([label,f"{row['validation']['perplexity']:.2f}",f"{metrics['perplexity']:.2f}",f"{100*metrics['top1']:.2f}",f"{100*metrics['top3']:.2f}",f"{100*metrics['top5']:.2f}",f"{row['training_seconds']:.2f}"])
    story.append(academic_table(['Model','Val. PPL','Test PPL','Top-1 %','Top-3 %','Top-5 %','Fit (s)'],rows,[128,61,61,56,56,56,WIDTH-418],styles))
    p('Table 2. Bold marks selection by validation NLL, not test accuracy. Fit time covers the retained configuration\'s fitting/checkpoint work, excluding the entire baseline search and final test scoring; timings depend on implementation and hardware.','Small')
    p(training_description(results))
    heading('5.1. Interpretation')
    gain=100*(unigram['test']['perplexity']-score['perplexity'])/unigram['test']['perplexity']
    direction='reduces' if gain>=0 else 'increases'
    p(f"The validation-selected <b>{escape(best['display_name'])}</b> {direction} test perplexity by {abs(gain):.1f}% relative to the unigram. "
      f"The selected HMM is {escape(best_hmm['display_name'])}: test perplexity {best_hmm['test']['perplexity']:.2f}, Top-3 accuracy {100*best_hmm['test']['top3']:.2f}%. "
      f"The selected RNN is {escape(best_rnn['display_name'])}: test perplexity {best_rnn['test']['perplexity']:.2f}, Top-3 accuracy {100*best_rnn['test']['top3']:.2f}%.")
    rank_winner=max(models,key=lambda row:row['test']['top3'])
    if rank_winner['name']!=best['name']:
        p(f"Among the frozen variants, {escape(rank_winner['display_name'])} has the highest test Top-3 accuracy ({100*rank_winner['test']['top3']:.2f}%), "
          f"compared with {100*score['top3']:.2f}% for the NLL-selected model. This descriptive observation does not change the validation-based selection.")
    p('NLL and Top-k accuracy measure different behavior: assigning more probability to the actual token can improve NLL without changing its displayed rank. '
      'Increasing hidden capacity need not improve held-out performance, since optimization and available data limit effective use of extra parameters. '
      'The findings describe the tested configurations and budgets, not a universal ranking of model families.')
    test=corpus['test']
    p(f"Test denominators are {test['tokens_including_eos']:,} encoded targets and {test['lexical_tokens']:,} lexical targets; {100*test['oov_rate']:.2f}% of lexical targets are unknown. "
      'Both metrics are needed to avoid treating high probability for the aggregated UNK symbol as successful completion.','Small')

    # Page 4: a single compact learning figure and qualitative comparisons.
    new_page()
    heading('6. Learning behavior and example completions')
    story.append(LearningFigure(best_hmm,best_rnn))
    p('Figure 1. Training and validation NLL for the selected HMM and RNN; the dot marks the retained checkpoint. '
      'The horizontal axes count different training operations and do not imply equal computation per step.','Small')
    p(checkpoint_description(best_hmm,best_rnn))
    hmm_rows=[row for row in models if row['family']=='HMM']
    ranges=[]
    for states in sorted({row['parameters']['n_states'] for row in hmm_rows}):
        values=[row['validation']['nll'] for row in hmm_rows if row['parameters']['n_states']==states]
        ranges.append(f'K={states}: {min(values):.3f}-{max(values):.3f}')
    p('HMM validation NLL ranges across initializations are '+'; '.join(ranges)+'. '
      'Two seeds provide a limited check of initialization sensitivity, not a reliable variance or significance estimate. RNN hidden sizes use one fixed seed.')
    heading('6.1. Illustrative predictions')
    examples=[]
    for example in results.get('examples',[])[:4]:
        predicted=[]
        for row in (best_count,best_hmm,best_rnn):
            values=example['predictions'].get(row['name'])
            if values is None: raise ValueError(f"Missing example predictions for {row['name']}; record each selected family.")
            predicted.append(escape(', '.join(item['word'] for item in values[:3])))
        examples.append([escape(example['context']),*predicted])
    story.append(academic_table(['Input context','Best count model','Best HMM','Best RNN'],examples,[WIDTH*.31,WIDTH*.23,WIDTH*.23,WIDTH*.23],styles))
    p('Table 3. Top three lexical suggestions for fixed demonstration prompts. Each family representative is chosen by validation NLL. '
      'These prompts illustrate interface behavior; they are not test metrics or evidence that a model understands the subject.','Small')
    p('The command-line interface accepts a text prefix and loads saved parameters without retraining. It can also filter suggestions by the beginning of an unfinished next word. '
      'Candidates retain their original probabilities; filtering the display does not alter the distribution used for perplexity.')

    # Page 5: scoped discussion, reproducibility, code URL, and references.
    new_page()
    heading('7. Discussion and limitations')
    p('This study demonstrates an end-to-end autocomplete pipeline with explicit latent-state inference and a trainable recurrent alternative. '
      'Local count models exploit recurring word combinations, HMMs share emissions through discrete states, and RNNs learn continuous recurrent features. '
      'The comparison concerns these implementations and training budgets, rather than proving that one architecture is intrinsically superior.')
    p('The corpus is one historical work combining philosophical exposition, psychological description, and quoted personal accounts. '
      'It does not represent all philosophy of religion or contemporary messages. Source-order splitting can shift topics between training and test; grouping paragraphs does not remove repetitions already present in the book. '
      'Unknown words cannot be suggested, casing and most punctuation are discarded, and sentence segmentation and footnote removal are approximate.')
    p('The study uses a small hyperparameter grid, two HMM seeds, one RNN seed, and one held-out split, reporting descriptive scores without confidence intervals. '
      'Future work should compare several books or authors, repeat neural training across seeds, vary vocabulary size, and measure typing effort saved. '
      'A pretrained comparison should disclose its additional data and computation.')
    heading('8. Reproducibility and code availability')
    if repository_url:
        p(f'Source code, preprocessing, configuration, and reproduction instructions are available at '
          f'<link href="{escape(repository_url)}" color="#000000">{escape(repository_url)}</link>. '
          'The accompanying archive contains the raw corpus, measured output, saved models, and this report.')
    else:
        p('The accompanying archive contains source code, raw corpus and provenance, configuration, measured output, saved models, tests, and this report. '
          'No public repository URL is asserted in this version.')
    p('Run from the project directory after creating and activating a Python environment:','Small')
    commands=('python -m pip install -r requirements-report.txt<br/>'
              'python predict.py --text "religious experience is" --top-k 5<br/>'
              'python -m unittest discover -s tests -v<br/>'
              'python experiment.py --output output_reproduced<br/>'
              'python scripts/build_report.py --results output_reproduced/results.json<br/>'
              '&nbsp;&nbsp;--output output_reproduced/project_report.pdf')
    if repository_url:
        commands+=f'<br/>&nbsp;&nbsp;--repository-url {escape(repository_url)}'
    p(commands,'Code')
    lines='three' if repository_url else 'two'
    p(f'Enter the last {lines} lines as one command. Raw data are bundled, so training needs no corpus download. '
      'Tests cover normalized probabilities, exact tiny-HMM likelihood and expected counts, vocabulary isolation, checkpoint restoration, persistence, and evaluation denominators. '
      f"Recorded environment: Python {escape(config['python'])}, NumPy {escape(config['numpy'])}, PyTorch {escape(str(config.get('torch','recorded in results')))}; additional platform details are saved in results.json.",'Small')
    heading('References')
    references=[
        '[1] W. James, <i>The Varieties of Religious Experience: A Study in Human Nature</i> (1902). Project Gutenberg eBook 621. <link href="https://www.gutenberg.org/ebooks/621" color="#000000">https://www.gutenberg.org/ebooks/621</link>. Exact source and digest recorded in corpus metadata.',
        '[2] Course notes, <i>Bayesian Inference</i>, 25 June 2025, sections 5.4-5.5, pp. 30-32 (HMMs and Baum-Welch), and section 5.7, pp. 33-34 (recurrent networks). Supplied with the assignment.',
        '[3] L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," <i>Proceedings of the IEEE</i> 77(2), 257-286 (1989). <link href="https://doi.org/10.1109/5.18626" color="#000000">doi:10.1109/5.18626</link>.',
        '[4] J. L. Elman, "Finding structure in time," <i>Cognitive Science</i> 14(2), 179-211 (1990). <link href="https://doi.org/10.1207/s15516709cog1402_1" color="#000000">doi:10.1207/s15516709cog1402_1</link>.',
    ]
    for reference in references: p(reference,'Small')
    output.parent.mkdir(parents=True,exist_ok=True)
    document=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=15*mm,bottomMargin=18*mm,
                              title='Next-Word Prediction in Philosophy of Religion',author=author,subject='N-gram, Hidden Markov and Recurrent Neural Models')
    document.build(story,onFirstPage=page_footer,onLaterPages=page_footer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'output/results.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/pdf/project_report.pdf')
    parser.add_argument('--repository-url',default='',help='Verified published GitHub repository URL to print.')
    parser.add_argument('--author',default='Avi Berezovski',help='Author name printed below the course title.')
    parser.add_argument('--student-id',default='',help='Optional private student ID. Requires a separate --output path.')
    parser.add_argument('--date',default='November 2026',help='Month and year printed in the title block.')
    args=parser.parse_args()
    if args.repository_url and not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?',args.repository_url):
        parser.error('--repository-url must be a full HTTPS GitHub repository URL.')
    if args.student_id and not re.fullmatch(r'[0-9]{5,12}',args.student_id):
        parser.error('--student-id must contain 5 to 12 digits.')
    if args.student_id and args.output.resolve()==(ROOT/'output/pdf/project_report.pdf').resolve():
        parser.error('Use a separate --output path for the private report containing a student ID.')
    build_report(read_experiment(args.results),args.output,args.repository_url,args.author,args.student_id,args.date)
    print(args.output)


if __name__=='__main__':
    main()
