#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: ldpc_encoder
# GNU Radio version: 3.10.9.2

from gnuradio import analog
from gnuradio import blocks
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




class ldpc_encoder(gr.top_block):

    def __init__(self, alist='/home/edd/dev/space-jam/3.0/alists/ccsds_tc_64.alist', k=64, n=128, n_frames=1, out_file='/tmp/out.bytes', random_frames=1, seed=0):
        gr.top_block.__init__(self, "ldpc_encoder", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.alist = alist
        self.k = k
        self.n = n
        self.n_frames = n_frames
        self.out_file = out_file
        self.random_frames = random_frames
        self.seed = seed

        ##################################################
        # Variables
        ##################################################
        self.puncturing = puncturing = ''

        ##################################################
        # Blocks
        ##################################################

        self.ldpc_toolbox_ldpc_encoder_0 = ldpc_toolbox.ldpc_encoder(alist, puncturing, n, k)
        self.blocks_vector_to_stream_0 = blocks.vector_to_stream(gr.sizeof_char*1, n)
        self.blocks_unpack_k_bits_bb_0 = blocks.unpack_k_bits_bb(8)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_char*1, k)
        self.blocks_head_0 = blocks.head(gr.sizeof_char*n, n_frames)
        self.blocks_file_sink_1 = blocks.file_sink(gr.sizeof_char*1, out_file, False)
        self.blocks_file_sink_1.set_unbuffered(False)
        self.analog_random_uniform_source_x_0 = analog.random_uniform_source_b(0, 256 if random_frames else 1, seed)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_uniform_source_x_0, 0), (self.blocks_unpack_k_bits_bb_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_vector_to_stream_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.ldpc_toolbox_ldpc_encoder_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_vector_to_stream_0, 0), (self.blocks_file_sink_1, 0))
        self.connect((self.ldpc_toolbox_ldpc_encoder_0, 0), (self.blocks_head_0, 0))


    def get_alist(self):
        return self.alist

    def set_alist(self, alist):
        self.alist = alist

    def get_k(self):
        return self.k

    def set_k(self, k):
        self.k = k

    def get_n(self):
        return self.n

    def set_n(self, n):
        self.n = n

    def get_n_frames(self):
        return self.n_frames

    def set_n_frames(self, n_frames):
        self.n_frames = n_frames
        self.blocks_head_0.set_length(self.n_frames)

    def get_out_file(self):
        return self.out_file

    def set_out_file(self, out_file):
        self.out_file = out_file
        self.blocks_file_sink_1.open(self.out_file)

    def get_random_frames(self):
        return self.random_frames

    def set_random_frames(self, random_frames):
        self.random_frames = random_frames

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed

    def get_puncturing(self):
        return self.puncturing

    def set_puncturing(self, puncturing):
        self.puncturing = puncturing



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--alist", dest="alist", type=str, default='/home/edd/dev/space-jam/3.0/alists/ccsds_tc_64.alist',
        help="Set alist [default=%(default)r]")
    parser.add_argument(
        "--k", dest="k", type=intx, default=64,
        help="Set k [default=%(default)r]")
    parser.add_argument(
        "--n", dest="n", type=intx, default=128,
        help="Set n [default=%(default)r]")
    parser.add_argument(
        "--n-frames", dest="n_frames", type=intx, default=1,
        help="Set n_frames [default=%(default)r]")
    parser.add_argument(
        "--out-file", dest="out_file", type=str, default='/tmp/out.bytes',
        help="Set out_file [default=%(default)r]")
    parser.add_argument(
        "--random-frames", dest="random_frames", type=intx, default=1,
        help="Set random_frames [default=%(default)r]")
    parser.add_argument(
        "--seed", dest="seed", type=intx, default=0,
        help="Set seed [default=%(default)r]")
    return parser


def main(top_block_cls=ldpc_encoder, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(alist=options.alist, k=options.k, n=options.n, n_frames=options.n_frames, out_file=options.out_file, random_frames=options.random_frames, seed=options.seed)

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
