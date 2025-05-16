#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Symbol Sync
# Author: edd
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation




class symbol_sync(gr.top_block):

    def __init__(self, TED_type='', constellation='digital.constellation_qpsk().base()', in_file='./data/out.iq', n_taps=11, out_file='./data/output.iq', rrc_excess_bandwidth=0.3, samp_rate=5000000, sps_estimate=8.65):
        gr.top_block.__init__(self, "Symbol Sync", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.TED_type = TED_type
        self.constellation = constellation
        self.in_file = in_file
        self.n_taps = n_taps
        self.out_file = out_file
        self.rrc_excess_bandwidth = rrc_excess_bandwidth
        self.samp_rate = samp_rate
        self.sps_estimate = sps_estimate

        ##################################################
        # Blocks
        ##################################################

        self.filter_fft_rrc_filter_0 = filter.fft_filter_ccc(1, firdes.root_raised_cosine(1, samp_rate, (samp_rate/sps_estimate), rrc_excess_bandwidth, 11), 1)
        if TED_type == "mueller":
            self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
                digital.TED_MUELLER_AND_MULLER,
                sps_estimate,
                0.045,
                1.0,
                1.0,
                1.5,
                1,
                eval(constellation),
                digital.IR_MMSE_8TAP,
                128,
                [])
        elif TED_type == "ml":
            self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
                digital.TED_SIGNUM_TIMES_SLOPE_ML,
                sps_estimate,
                0.045,
                1.0,
                1.0,
                1.5,
                1,
                eval(constellation),
                digital.IR_MMSE_8TAP,
                128,
                [])
        else:
            raise NotImplementedError()

        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, in_file, False, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, out_file, False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.filter_fft_rrc_filter_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.filter_fft_rrc_filter_0, 0), (self.digital_symbol_sync_xx_0, 0))


    def get_TED_type(self):
        return self.TED_type

    def set_TED_type(self, TED_type):
        self.TED_type = TED_type

    def get_constellation(self):
        return self.constellation

    def set_constellation(self, constellation):
        self.constellation = constellation

    def get_in_file(self):
        return self.in_file

    def set_in_file(self, in_file):
        self.in_file = in_file
        self.blocks_file_source_0.open(self.in_file, False)

    def get_n_taps(self):
        return self.n_taps

    def set_n_taps(self, n_taps):
        self.n_taps = n_taps

    def get_out_file(self):
        return self.out_file

    def set_out_file(self, out_file):
        self.out_file = out_file
        self.blocks_file_sink_0.open(self.out_file)

    def get_rrc_excess_bandwidth(self):
        return self.rrc_excess_bandwidth

    def set_rrc_excess_bandwidth(self, rrc_excess_bandwidth):
        self.rrc_excess_bandwidth = rrc_excess_bandwidth
        self.filter_fft_rrc_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps_estimate), self.rrc_excess_bandwidth, 11))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle2_0.set_sample_rate((self.samp_rate/10))
        self.filter_fft_rrc_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps_estimate), self.rrc_excess_bandwidth, 11))

    def get_sps_estimate(self):
        return self.sps_estimate

    def set_sps_estimate(self, sps_estimate):
        self.sps_estimate = sps_estimate
        self.digital_symbol_sync_xx_0.set_sps(self.sps_estimate)
        self.filter_fft_rrc_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps_estimate), self.rrc_excess_bandwidth, 11))



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--TED-type", dest="TED_type", type=str, default='',
        help="Set TED_type [default=%(default)r]")
    parser.add_argument(
        "--constellation", dest="constellation", type=str, default='digital.constellation_qpsk().base()',
        help="Set constellation [default=%(default)r]")
    parser.add_argument(
        "--in-file", dest="in_file", type=str, default='./data/out.iq',
        help="Set in_file [default=%(default)r]")
    parser.add_argument(
        "--n-taps", dest="n_taps", type=intx, default=11,
        help="Set n_taps [default=%(default)r]")
    parser.add_argument(
        "--out-file", dest="out_file", type=str, default='./data/output.iq',
        help="Set out_file [default=%(default)r]")
    parser.add_argument(
        "--rrc-excess-bandwidth", dest="rrc_excess_bandwidth", type=eng_float, default=eng_notation.num_to_str(float(0.3)),
        help="Set rrc_excess_bandwidth [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=intx, default=5000000,
        help="Set samp_rate [default=%(default)r]")
    parser.add_argument(
        "--sps-estimate", dest="sps_estimate", type=eng_float, default=eng_notation.num_to_str(float(8.65)),
        help="Set sps_estimate [default=%(default)r]")
    return parser


def main(top_block_cls=symbol_sync, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(TED_type=options.TED_type, constellation=options.constellation, in_file=options.in_file, n_taps=options.n_taps, out_file=options.out_file, rrc_excess_bandwidth=options.rrc_excess_bandwidth, samp_rate=options.samp_rate, sps_estimate=options.sps_estimate)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
