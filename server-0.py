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




app = Flask(__name__)
#flask_cors.CORS(app)
optL = [True]
Dict = {}
#bpe = applybpe.initializebpe("tools/2M.bpedict.zh")

'''

def main(opt):
    freedecoding = [True]
    pre_tgt      = [True]


    font_align = ('Courier', 20)# 'Roboto Mono''Letter Gothic'#font_align
    font_label = ('Times',14)

'''




@app.route('/')
def my_form():
    return render_template('interface.html', model=optL[0].model, waitnum=optL[0].model[1])



@app.route('/start')
def stream():
	#main(optL[0])
    #a = request.args.get('a', 0, type=int)
    return jsonify(model=optL[0].model)


'''
@app.route('/fillScreen')
def fillScreen():
    return jsonify( lines=n_sent,
    				BoxDisplay_chn=chnDisplay,
    				BoxDisplay_eng=resDisplay,
    				BoxDisplay_baseline=baselineDisplay)
'''

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
    #onmt.opts.add_md_help_argument(parser)
    #onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    print(opt.model)
    logger = init_logger(opt.log_file)    
    optL[0] = opt


    app.logger.info('message processed')
    #app.run(host='0.0.0.0') #, debug=True)
    app.run(debug=True)
