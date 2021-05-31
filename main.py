import pdftotext
import textract
import re
import nltk
import glob
import os
import spacy
import datetime
import pandas as pd
from itertools import combinations
from gensim import corpora
from gensim import models
from spacy.matcher import Matcher
from nltk import tokenize
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup, NavigableString
from multiprocessing import Pool

nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer(language='english')


def sent_trim(string):
    string = re.sub(r'\r', ' ', string)
    string = re.sub(r'\n', ' ', string)
    string = re.sub(r'[-_]+', ' ', string)
    string = re.sub(r' +', ' ', string)
    string = re.sub(r'^ *([oV] )', ' ', string)
    string = re.sub(r'^[^\w.,!?%:/()&@;-]+', '', string)
    string = re.sub(r'[^\w.,!?%:/()）。&@;]+$', '', string)
    return string


def text_split(text):
    replace_dict = {
        ' Responsibilities  ': '                   ',
        '   Job': '      ',
        u'\xa0': u' ',
        '\r': '\n',
        '\t': ' '
    }
    for key, value in replace_dict.items():
        text = text.replace(key, value)
    # text = re.sub(r'_+', ' ', text)
    # text = re.sub(r' +', ' ', text)
    sent_list = [sent for sent in text.split('\n') if sent]
    sent_list = [sent_trim(sent) for sent in sent_list]
    return [sent for sent in sent_list if sent and (sent != 'Public' or sent != 'Open')]


def text_optimize(sent_list):
    # if the sentence ends with preposition/conjunction/articles/tos
    end_tags_list = ['IN', 'CC', 'DT', 'TO']
    # Verbs and adjectives
    end_tags_alt_list = ['VB', 'JJ']
    end_pun_list = ['.', ',', ';']
    for i in reversed(range(1, len(sent_list))):
        sent = sent_list[i]
        prv_sent = sent_list[i - 1]
        prv_sent_tagged = nltk.pos_tag(tokenize.word_tokenize(prv_sent))
        # if (sent[-1] == '.' and len(sent.split(' ')) <= 1)
        if (len(sent.split(' ')) <= 1) or \
                (sent[0].islower() and any([sent[-1] == pun for pun in end_pun_list])) or \
                any([prv_sent_tagged[-1][1] == tag for tag in end_tags_list]) or \
                (sent[-1] == '.' and any([tag in prv_sent_tagged[-1][1] for tag in end_tags_alt_list])):
            sent_list[i - 1] = ' '.join([prv_sent, sent_list.pop(i)])

    for i in reversed(range(1, len(sent_list) - 1)):
        sent = sent_list[i]
        prv_sent = sent_list[i - 1]
        next_sent = sent_list[i + 1]
        if (sent[0].islower() and prv_sent[0].isupper() and next_sent[0].isupper()) or \
                (prv_sent[0].isdigit() and next_sent[0].isdigit() and abs(int(next_sent[0]) - int(prv_sent[0])) == 1):
            sent_list[i - 1] = ' '.join([prv_sent, sent_list.pop(i)])
    return sent_list


def parse_html(html):
    def check_append(_tag):
        if _tag.find('br'):
            text_list.extend([sent_trim(string) for string in _tag.stripped_strings])
        else:
            text_list.append(sent_trim(_tag.text))

    soup = BeautifulSoup(html, 'html5lib')
    for match in soup.findAll(['font', 'a', 'b', 'o:p', 'span', 'u', 'i']):
        match.unwrap()
    if len(soup.body.contents) == 1 and soup.body.find('div'):
        soup.body.div.unwrap()

    text_tags = soup.findAll(['div', 'li', 'p'])
    text_list = []
    for tag in text_tags:
        if tag.name == 'li':
            if tag.find('p'):
                if not re.sub(r'\s|\xa0', '', tag.find('p').text):
                    check_append(tag)
            else:
                check_append(tag)
        elif tag.name == 'div':
            if tag.parent.name != 'li':
                if tag.find('ul') and isinstance(tag.contents[0], NavigableString):
                    text_list.append(sent_trim(tag.contents[0]))
                else:
                    check_append(tag)
        else:
            check_append(tag)

    for i, text in enumerate(text_list):
        if text:
            if text[0].islower() and text_list[i - 1][0].isnumeric():
                text_list[i - 1] = ' '.join([text_list[i - 1], text])
                text_list[i] = ''

    text_list = [text for text in text_list if text]
    return text_list


def tokenizer(sent):
    doc = nlp(sent)
    tokens_raw = [token for token in doc if token.is_alpha and not token.is_stop and len(token.text) > 1]
    nouns_tokens_raw = [token for token in tokens_raw if token.pos_ == 'PROPN' or token.pos_ == 'NOUN']
    tokens_lemmatized = [
        {'stem': stemmer.stem(token.lemma_.lower()), 'raw': token.text} for token in tokens_raw]
    nouns_tokens_lemmatized = [
        {'stem': stemmer.stem(token.lemma_.lower()), 'raw': token.text} for token in nouns_tokens_raw]
    return {'nouns': nouns_tokens_lemmatized, 'all': tokens_lemmatized}


# def text_optimize2(sent_list):
#     for i in reversed(range(len(sent_list))):
#         sent = sent_list[i]
#         prv_sent = sent_list[i - 1]
#         if sent and i != 0:
#             if sent[-1] == '.' and not sent[0].isdigit():
#                 if '. ' in prv_sent[3:-1]:
#                     prv_sent_list = tokenize.sent_tokenize(prv_sent)
#                     sent_list[i] = ' '.join([prv_sent_list[-1], sent])
#                     sent_list[i - 1] = ' '.join(prv_sent_list[:-1])
#     # split even e.g. and ltd.
#     # return [sent_split for sent in sent_list for sent_split in tokenize.sent_tokenize(sent) if sent_split]
#     return [sent for sent in sent_list if sent]


def cv_read_tokenize(row_dict):
    path = row_dict["resume"]
    ext = os.path.splitext(path)[1]
    if ext == '.pdf':
        with open(path, 'rb') as f:
            pdf = pdftotext.PDF(f)
        text = '\n'.join([*pdf])
    else:
        try:
            text = textract.process(path).decode('utf-8-sig')
        except textract.exceptions.ShellError:
            print(f'{path} is not supported.')
            os._exit(1)
    doc = nlp(text)
    years = [x.text for x in doc if x.pos_ == 'NUM']
    years = [int(y) for x in years for y in re.findall(r'\b(?:19|20)\d{2}\b', x)]
    exp = max(years) - min(years) if years else -1

    # uni
    matcher = Matcher(nlp.vocab)
    matched_sents = []

    def collect_sents(matcher, doc, i, matches):
        _, start, end = matches[i]
        span = doc[start:end]
        trimmed_start = max(span.sent.start, span.start - 4)
        trimmed_end = min(span.sent.end, span.end + 4)
        matched_sents.append(sent_trim(doc[trimmed_start:trimmed_end].text))

    matcher.add(
        "uni", collect_sents,
        [{"LOWER": "university"}], [{"LOWER": "universiti"}], [{"LOWER": "college"}], [{"LOWER": "institute"}],
        [{"LOWER": "universität"}], [{"LOWER": "universitaet"}], [{"LOWER": "universidad"}],
        [{"LOWER": "universiteit"}], [{"LOWER": "université"}]
    )
    matcher(doc)

    cv = text_optimize(text_split(text))
    return [{'text': sent, 'tokens': tokenizer(sent)} for sent in cv] if cv else [], exp, ';'.join(matched_sents)


def calculate_tfidf(jf_group_df):
    bow = [
        [
            tokens['stem']
            for sentence in row['resume']
            for tokens in sentence['tokens']['all']
        ]
        for _, row in jf_group_df.iterrows() if row['resume']
    ]
    dictionary = corpora.Dictionary(bow)
    corpus = [dictionary.doc2bow(text) for text in bow]
    return {
        dictionary.get(token_id): score for doc in
        models.TfidfModel(corpus)[corpus] for token_id, score in doc
    }


def find_and_filter(cv_id):
    files = glob.glob(f'data/CVs/{cv_id}_*[(.pdf|.doc|.docx)]')
    if files:
        re_cv = re.compile(r'cv|resume|c\.v|c\.v\.|résumé|lebenslauf|简历', flags=re.IGNORECASE)
        files = [*filter(lambda x: re_cv.search(os.path.basename(x)), files)]
        if len(files) > 1:
            re_chi = re.compile(r'ch|chn|chi|chinese', flags=re.IGNORECASE)
            files = [*filter(lambda x: not re_chi.search(os.path.basename(x)), files)]
        if len(files) > 1:
            re_en = re.compile(r'en|english|英', flags=re.IGNORECASE)
            files = [*filter(lambda x: re_en.search(os.path.basename(x)), files)]
    return files[0] if files else ''


def jd_tokenize(x):
    return [{'text': sent, 'tokens': tokenizer(sent)} for sent in parse_html(x)]


if __name__ == "__main__":
    with Pool() as p:
        try:
            jds_df = pd.read_csv(max(glob.glob('data/JDs/JDs*'), key=os.path.getctime), dtype=str)
        except ValueError:
            print('Please mount folder and check if JDs file exist in JDs folder!')
            os._exit(1)
        jds_df = jds_df[['JobField', 'Requisition_ID', 'Requisition_Title', 'JD_English']].drop_duplicates()
        cvs_df = pd.concat(
            [pd.read_csv(max(glob.glob('data/CVs_candidate/cv_candidate_app*'), key=os.path.getctime), dtype=str),
             pd.read_csv(max(glob.glob('data/CVs_candidate/cv_candidate_pro*'), key=os.path.getctime), dtype=str)]
        ).drop_duplicates()
        cvs_df = cvs_df.astype(str)

        # join to save calculation power for the next operation
        cvs_df = cvs_df.groupby(
            [x for x in cvs_df.columns if x != 'Requisition_ID']
        )['Requisition_ID'].apply(list).reset_index()
        cvs_df['resume'] = p.map(find_and_filter, [*cvs_df['Candidate_ID']])
        cvs_df = cvs_df[cvs_df['resume'] != '']

        # eliminate cvs that has less than two in a job field
        filtered_df = cvs_df.explode('Requisition_ID').merge(
            jds_df, left_on='Requisition_ID', right_on='Requisition_ID'
        ).groupby('JobField').filter(
            lambda x: len(x) > 1
        )
        # Separate again for groupby cvs_df only
        cvs_df = filtered_df.drop(['JD_English', 'JobField', 'Requisition_Title'], axis=1)
        jds_df = jds_df[jds_df['Requisition_ID'].isin(filtered_df['Requisition_ID'])]

        # tokenize jds
        jds_df['JD_English'] = p.map(jd_tokenize, [*jds_df['JD_English']])

        # compress cvs_df again to save computation power for tokenize
        cvs_df = cvs_df.groupby(
            [x for x in cvs_df.columns if x != 'Requisition_ID']
        )['Requisition_ID'].apply(list).reset_index()
        # read resume and tokenize
        cvs_df['resume'], cvs_df['experience'], cvs_df['university'] = zip(
            *p.map(cv_read_tokenize, cvs_df.to_dict('records')))
        # explode back to full
        cvs_df = cvs_df.explode('Requisition_ID').merge(
            jds_df[['Requisition_ID', 'Requisition_Title', 'JobField']],
            left_on='Requisition_ID', right_on='Requisition_ID'
        )

        # tfidf
        tfidf_dict = {
            job_field: calculate_tfidf(job_field_df)
            for job_field, job_field_df in cvs_df.groupby('JobField')
        }
        jd_terms_df = pd.DataFrame(
            [[
                row['JobField'],
                row['Requisition_ID'],
                row['Requisition_Title'],
                sent['text'],
                terms[0]['stem'],
                terms[1]['stem'],
                terms[0]['raw'],
                terms[1]['raw']
            ]
                for _, row in jds_df.iterrows()
                for sent in row['JD_English']
                for terms in combinations(sent['tokens']['nouns'], 2)
                if terms[0]['stem'] != terms[1]['stem']],
            columns=[
                'JobField',
                'Requisition_ID',
                'Requisition_Title',
                'sentence_text',
                'term1_stem',
                'term2_stem',
                'term1_raw',
                'term2_raw'
            ])
        cv_terms_df = pd.DataFrame(
            [[
                job_field,
                row['Candidate_ID'],
                row['FirstName'],
                row['LastName'],
                row['EmailAddress'],
                row['Gender_Code'],
                row['Requisition_ID'],
                row['Requisition_Title'],
                row['experience'],
                row['university'],
                sent['text'],
                terms[0]['stem'],
                terms[1]['stem'],
                terms[0]['raw'],
                terms[1]['raw'],
                tfidf_dict[job_field].get(terms[0]['stem'], 0),
                tfidf_dict[job_field].get(terms[1]['stem'], 0)
            ]
                for job_field, job_field_df in cvs_df.groupby('JobField')
                for _, row in job_field_df.iterrows()
                for sent in row['resume']
                for terms in combinations(sent['tokens']['nouns'], 2)
                if terms[0]['stem'] != terms[1]['stem']],
            columns=[
                'JobField',
                'Candidate_ID',
                'FirstName',
                'LastName',
                'EmailAddress',
                'Gender_Code',
                'Requisition_ID',
                'Requisition_Title',
                'experience',
                'university',
                'sentence_text',
                'term1_stem',
                'term2_stem',
                'term1_raw',
                'term2_raw',
                'term1_tf-idf',
                'term2_tf-idf'
            ])

        keywords_df = jd_terms_df.merge(cv_terms_df,
                                        left_on=['JobField', 'term1_stem', 'term2_stem'],
                                        right_on=['JobField', 'term1_stem', 'term2_stem'], suffixes=('_jd', '_cv'))
        keywords_df['terms_stem'] = keywords_df[['term1_stem', 'term2_stem']].apply(lambda x: ' '.join(x), axis=1)
        keywords_df['terms_raw_jd'] = keywords_df[['term1_raw_jd', 'term2_raw_jd']].apply(lambda x: ' '.join(x), axis=1)
        keywords_df['terms_raw_cv'] = keywords_df[['term1_raw_cv', 'term2_raw_cv']].apply(lambda x: ' '.join(x), axis=1)
        keywords_df['terms_tf-idf'] = keywords_df['term1_tf-idf'] + keywords_df['term2_tf-idf']
        keywords_df['total_tf-idf_per_cv'] = keywords_df.groupby('Candidate_ID')['terms_tf-idf'].transform('sum')
        today = str(datetime.date.today())
        keywords_df['process_date'] = today
        if not os.path.exists('data/OUTPUT'):
            os.makedirs('data/OUTPUT')
        keywords_df.to_csv(f'data/OUTPUT/{today}.csv', index=False, encoding='utf-8-sig', sep='|')
        print('Done.')
