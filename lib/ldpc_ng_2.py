#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: ldpc_ng_2
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import ldpc_toolbox
import numpy as np




class ldpc_ng_2(gr.top_block):

    def __init__(self, alist='/home/edd/dev/space-jam/3.0/alists/dvbs2_5_6_short.alist', in_file='/tmp/infile', k=13320, max_iterations=100, n=16200, out_file='/tmp/out.bytes', decoder="Phif64"):
        gr.top_block.__init__(self, "ldpc_ng_2", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.alist = alist
        self.in_file = in_file
        self.k = k
        self.max_iterations = max_iterations
        self.n = n
        self.out_file = out_file

        ##################################################
        # Variables
        ##################################################
        self.puncturing = puncturing = ''

        ##################################################
        # Blocks
        ##################################################

        self.ldpc_toolbox_ldpc_decoder_0 = ldpc_toolbox.ldpc_decoder(alist, decoder, puncturing, n, k, max_iterations)
        self.blocks_file_source_1 = blocks.file_source(gr.sizeof_float*n, in_file, False, 0, 0)
        self.blocks_file_source_1.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_char*k, out_file, False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_1, 0), (self.ldpc_toolbox_ldpc_decoder_0, 0))
        self.connect((self.ldpc_toolbox_ldpc_decoder_0, 0), (self.blocks_file_sink_0, 0))


    def get_alist(self):
        return self.alist

    def set_alist(self, alist):
        self.alist = alist

    def get_in_file(self):
        return self.in_file

    def set_in_file(self, in_file):
        self.in_file = in_file
        self.blocks_file_source_1.open(self.in_file, False)

    def get_k(self):
        return self.k

    def set_k(self, k):
        self.k = k

    def get_max_iterations(self):
        return self.max_iterations

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations

    def get_n(self):
        return self.n

    def set_n(self, n):
        self.n = n

    def get_out_file(self):
        return self.out_file

    def set_out_file(self, out_file):
        self.out_file = out_file
        self.blocks_file_sink_0.open(self.out_file)

    def get_puncturing(self):
        return self.puncturing

    def set_puncturing(self, puncturing):
        self.puncturing = puncturing



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--alist", dest="alist", type=str, default='/home/edd/dev/space-jam/3.0/alists/dvbs2_5_6_short.alist',
        help="Set alist [default=%(default)r]")
    parser.add_argument(
        "--in-file", dest="in_file", type=str, default='/tmp/infile',
        help="Set in_file [default=%(default)r]")
    parser.add_argument(
        "--k", dest="k", type=intx, default=13320,
        help="Set k [default=%(default)r]")
    parser.add_argument(
        "--max-iterations", dest="max_iterations", type=intx, default=100,
        help="Set max_iterations [default=%(default)r]")
    parser.add_argument(
        "--n", dest="n", type=intx, default=16200,
        help="Set n [default=%(default)r]")
    parser.add_argument(
        "--out-file", dest="out_file", type=str, default='/tmp/out.bytes',
        help="Set out_file [default=%(default)r]")
    return parser


def main(top_block_cls=ldpc_ng_2, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(alist=options.alist, in_file=options.in_file, k=options.k, max_iterations=options.max_iterations, n=options.n, out_file=options.out_file)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
