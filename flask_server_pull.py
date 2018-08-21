#!/usr/bin/python
# coding-utf8

'''
Developer: Kaibo(lrushx)
Email: liukaib@oregonstate.edu
Created Date: Aug 15, 2018
'''

from __future__ import division, unicode_literals
from flask import Flask, jsonify, render_template, request
#import flask_cors
from werkzeug.utils import secure_filename 
import os
import time
import re
import json
import psutil



import argparse

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts
import tools.apply_bpe as applybpe
import jieba



# last version, make_translator2 is for baseline
#from onmt.translate.Translator import make_translator,make_translator2
#import onmt.io
#import onmt.ModelConstructor
from tkinter import *
import pinyin
#import torch
#from torch.autograd import Variable
import math
#import speech_recognition as sr
import operator

from micro_stream import MicrophoneStream
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from itertools import chain

from googletrans import Translator as word_trans

import time

app = Flask(__name__)
#flask_cors.CORS(app)
optL = [True]
Dict = {'chnBase':"",'engBase':"",'bslBase':"",
        'chnLine':"",'engLine':"",'bslLine':"",'pinyin':"",'word':"",        
        'chnLines':0,'engLines':0,'bslLines':0}
        #'chnWid':0,  'engWid':0,  'bslWid':0}

bpe = applybpe.initializebpe("tools/2M.bpedict.zh")
modelStatus = ["model loading..."]
MaxLine = 7
MaxWid  = 28
stream_end = [False]
duration = [0,0]

def main(opt):
    freedecoding = [True]
    pre_tgt      = [""]


    #font_align = ('Courier', 20)# 'Roboto Mono''Letter Gothic'#font_align
    #font_label = ('Times',14)

    translator = build_translator(opt, report_score=True)
    word_translator = word_trans()



    def alignment(zh_list,pinyin_list):
        zh = []
        py = []
        for i,j in zip(zh_list,pinyin_list):
            zh_len, pin_len = len(i)*2, len(j)
            len_diff = math.ceil((pin_len - zh_len)*2)
            i = i+" "*len_diff
            zh.append(i)
            py.append(j)
            
        return " ".join(zh)+"\n"+" ".join(py)
    def count_space(l1,l2,l3):
        ratio = 2
        l1 = int(l1 * ratio)
        w = max(l1,l2,l3) + 1
        if w & 1: w += 1
        return w-l1, w-l2, w-l3


    def trans_whole(src_txt_list, finishall=False):

        # 3 lines of Chn, pinyin, word_trans, with alignment
        pinyin_list = [pinyin.get(i) for i in src_txt_list]
        out_word_list = [word_translator.translate(i,dest='en').text.lower() for i in src_txt_list]
        
        '''
        BLANK = ' '
        chn_str, pinyin_str, word_str = '','',''
        for a,b,c in zip(src_txt_list, pinyin_list, out_word_list):
            if ord(a[0]) > 122:
                x,y,z = count_space(len(a), len(b), len(c))
                chn_str     += a + chr(12288)*(x>>1) #chr(12288)
                pinyin_str  += b + BLANK*y
                word_str    += c + BLANK*z
            else:
                chn_str     += a + '\t'
                pinyin_str  += b + '\t'
                word_str    += c + '\t'
        '''
        #pinyin_str = pinyin_str.replace('。','.')
        #word_str   = word_str.replace('。','.')

        chn_str = ' '.join(src_txt_list)
        pinyin_str = ' '.join(pinyin_list)
        word_str = ' '.join(out_word_list)

        # end of 3 lines

        src_txtidx_list = []
        res_baseline = ""
        res = ""
        # last version
        #src_txtidx_list = [translator.fields["src"].vocab.stoi[x] for x in src_txt_list]
        #translated_sent = translator.translate_batch(src_txtidx_list,translator.model_opt.waitk) # input is word id list
        
        # 8/8/2018 version
        if len(src_txt_list) >= translator.waitk:            
            current_input_bpe_lst = bpe.segment(" ".join(src_txt_list)).split()            
            src_txtidx_list = []
            src_txtidx_list = [translator.fields["src"].vocab.stoi[wordi] for wordi in current_input_bpe_lst]

            ### detect the end of sentence
            #if sentwidx == len(sampleWholeInput)-1:
            translated_sent, pre_tgt[0] = translator._translate_batch(src_txtidx_list,pre_tgt[0],freedecoding[0],finishall=finishall)
            print('#'*10, current_input_bpe_lst)
            print('#'*20, translated_sent)


            res = clean_eos(translated_sent)

            if finishall:
                res_baseline_list, pre_tgt[0] = translator._translate_batch(src_txtidx_list,pre_tgt[0],freedecoding=True,finishall=True)
                #res_baseline = " ".join(clean_eos(res_baseline_list)).replace("@@ ","") + '.'
                res_baseline = clean_eos(res_baseline_list)
                #pinyin_str += '。'
                #word_str += '.'
            freedecoding[0] = False            

        #print('temp: {} {} {} [{} {}] {}'.format(src_txt_list, res, src_txtidx_list, freedecoding[0], finishall, pre_tgt[0]))
        #print('\rChinese: {}'.format(chn_str), end = '')
        if finishall:
            print('\n\nChinese: {}'.format(chn_str))
            print('Pinyin: {}\nW-Trans:{}\nk-waits:{}\ngreedy: {}\n{}'.format(pinyin_str, word_str, 
                                                                                res, res_baseline, "".join(src_txt_list)))

        return chn_str, pinyin_str, word_str, res, res_baseline


    def clean_eos(s):
        if len(s)>=2 and s[-1] == "</s>":       # replace tail ".</s>" or "</s>" to "."
            if s[-2] == ".":
                s = s[:-1]
            else:
                s[-1] = "."
        #res = " ".join(translated_sent)
        return " ".join(s).replace("@@ ","")

    ## kaibo: flush current line for next sentence with additional new word, until printing the final transcript if `is_final`
    def listen_print_loop(responses):
        """Iterates through server responses and prints them.

        The responses passed is a generator that will block until a response
        is provided by the server.

        Each response may contain multiple results, and each result may contain
        multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
        print only the transcription for the top alternative of the top result.

        In this case, responses are provided for interim results as well. If the
        response is an interim one, print a line feed at the end of it, to allow
        the next result to overwrite it, until the response is a final one. For the
        final one, print a newline to preserve the finalized transcription.
        """

        
        num_words = 0   # number of words in the whole sentence (maybe multi lines)
        num_chars = 0   # number of char in current line
        #num_chars_printed = 0
        base_txt_list, txt_list = [], None
        #chn_str, pinyin_str, word_str = '','',''
        stream_end[0] = False
        Dict['chnLine'] = ""    
        Dict['pinyin'] = ""    
        Dict['word'] = ""  

        for response in responses:
            if not response.results:
                
                continue
            stream_end[0] = False
            # The `results` list is consecutive. For streaming, we only care about
            # the first result being considered, since once it's `is_final`, it
            # moves on to considering the next utterance.
            asr_result = response.results[0]
            if not asr_result.alternatives:
                continue

            # Display the transcription of the top alternative.
            transcript = asr_result.alternatives[0].transcript

            #print('+'*20,transcript, stream_end[0])

            if len(transcript) > num_chars and ord(transcript[-1]) > 128:   # kaibo: cut only if characters increases & not eng
                num_chars = len(transcript)
                transcript_cut = jieba.cut(transcript, cut_all=False)
                #txt_list = list(w.lower() if ord(w[0])<128 else w for w in chain(base_txt_list, transcript_cut))
                txt_list = list(map(lambda x: x.lower(), chain(base_txt_list, transcript_cut)))
                if len(txt_list) > num_words: # kaibo: update GUI only if words increases
                    num_words = len(txt_list)
                    if num_words > 1:
                        #chn_str, pinyin_str, word_str = trans(txt_list[:-1], chn_str, pinyin_str, word_str)
                        Dict['chnLine'], Dict['pinyin'], Dict['word'], Dict['engLine'], Dict['bslLine'] = trans_whole(txt_list[:-1])
            #print('-'*20, num_chars, num_words, Dict['chnLine'])
         
            if asr_result.is_final:
                #trans(txt_list + ['。'], chn_str, pinyin_str, word_str)
                #trans_whole(txt_list + ['。'])
                Dict['chnLine'], Dict['pinyin'], Dict['word'], eng, bsl = trans_whole(txt_list, finishall=True)
                
                Dict['chnLine'] += '。'
                Dict['pinyin'] += '.'
                Dict['word'] += '.'

                #update_base('chn', Dict['chnLine'].replace('\t','').replace(chr(12288),''), True)
                update_base('chn', Dict['chnLine'].replace(' ',''), True)
                update_base('eng', eng)
                update_base('bsl', bsl)

                stream_end[0] = True

                #stream_end[0] = False
                freedecoding[0] = True   # kaibo: added for 8/8/2018 version
                pre_tgt[0] = []         # kaibo: added for 8/8/2018 version
                src_txt_list = []                
                num_words = 0   # number of words in the whole sentence (maybe multi lines)
                num_chars = 0   # number of char in current line
                base_txt_list, txt_list = [], None                
                #print(". [end]")


    def update_base(key, s, ischn=False):
        s1, n = lineBreak(s, ischn=ischn)
        Dict[key+'Base'] += s1
        Dict[key+'Lines'] += n
        Dict[key+'Line'] += ""
        
        #print('{}:{}'.format(key+'Base', Dict[key+'Base']))


    
    # Audio recording parameters
    RATE = 16000
    #RATE = 10000
    CHUNK = int(RATE / 10)  # 100ms
    language_code = "cmn-Hans-CN"
    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    #stream_end[0] = False
    print('say something until a pause as ending in 65 seconds')
    modelStatus[0] = "model loaded sucessfully, say something within"
    duration[0] = time.time()
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)
        # Now, put the transcription responses to use.
        listen_print_loop(responses)
  



def lineBreak(s,ischn=False):
    MaxWid = 27 if ischn else 60
    wid, lines = 0, 0 
    ret = ''
    while wid+MaxWid <= len(s):
        p = wid+MaxWid
        if not ischn:
            while s[p] != ' ': p -= 1
        ret += (s[wid:p]+'<br>')
        wid =  p
        lines += 1
    if wid <= len(s):
        ret += (s[wid:]+'<br>')
        lines += 1
    #print('-'*20,ret)
    return ret+'<br>', lines+1


@app.route('/')
def interface():
    return render_template('interface.html', model=optL[0].model, waitnum=optL[0].model[1])



@app.route('/start')
def stream():
    Dict['chnLine'] = ""
    Dict['pinyin'] = ""
    Dict['word'] = ""
    Dict['chnBase'] = ""
    Dict['engBase'] = ""
    Dict['bslBase'] = ""
    Dict['chnLines'] = 0
    Dict['engLines'] = 0
    Dict['bslLines'] = 0
    modelStatus[0] = "model loading..."
    duration[0] = time.time()
    main(optL[0])
    #a = request.args.get('a', 0, type=int)
    return jsonify(model=optL[0].model)



@app.route('/fillScreen')
def fillScreen():
    chnBreak = engBreak = bslBreak = ""
    chnNewLines = engNewLines = bslNewLines = 0

    if not stream_end[0]:
        #chnBreak, chnNewLines = lineBreak(Dict['chnLine'].replace('\t','').replace(chr(12288),''),True)
        chnBreak, chnNewLines = lineBreak(Dict['chnLine'].replace(' ',''),True)
        engBreak, engNewLines = lineBreak(Dict['engLine'])
        bslBreak, bslNewLines = lineBreak(Dict['bslLine'])
    
    #print('!'*10, Dict['chnLines'], Dict['chnLines']+chnNewLines-MaxLine, type(Dict['chnLines']+chnNewLines-MaxLine))
    return jsonify( duration=time.time()-duration[0],
                    modelStatus=modelStatus[0],
                    scroll_chn=Dict['chnLines']+chnNewLines-MaxLine,
                    scroll_eng=Dict['engLines']+engNewLines-MaxLine,
                    scroll_bsl=Dict['bslLines']+bslNewLines-MaxLine,
                    Line_chn=Dict['chnLine'],
                    Line_pyn=Dict['pinyin'],
                    Line_wrd=Dict['word'],
                    BoxDisplay_chn=Dict['chnBase']+chnBreak,
                    BoxDisplay_eng=Dict['engBase']+engBreak,
                    BoxDisplay_bsl=Dict['bslBase']+bslBreak)


@app.route('/cpu', methods= ['GET'])
def get_cpu():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    t   = time.asctime() 
    return jsonify(cpu=cpu, ram=ram, time=t)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    print(opt.model)
    logger = init_logger(opt.log_file)    
    optL[0] = opt


    app.logger.info('message processed')
    #app.run(host='0.0.0.0') #, debug=True)
    app.run(debug=True)
