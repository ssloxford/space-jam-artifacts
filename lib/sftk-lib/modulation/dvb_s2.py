import numpy as np

from lib.modulation import PSK, APSK

# DVB-S2 constellations
# Defined in https://www.etsi.org/deliver/etsi_en/302300_302399/30230702/01.02.01_60/en_30230702v010201p.pdf

# PSK

BPSK = PSK(2)
QPSK = PSK(4)
PSK_8 = PSK(8)

# APSK
APSK_8_rate_100_180 = APSK([[0, 4], [6, 2, 1, 5], [7, 3]], [1, 5.32, 6.8], [0, 0.352*np.pi, 0])
APSK_8_rate_104_180 = APSK([[0, 4], [6, 2, 1, 5], [7, 3]], [1, 6.39, 8.0], [0, 0.352*np.pi, 0])

_apsk_16_labels = list(map(lambda x: list(map(lambda y: int(y, 2), x)), [['1100', '1110', '1111', '1101'], ['0100', '0000', '1000', '1010', '0010', '0110', '0111', '0011', '1011', '1001', '0001', '0101']]))
APSK_16_rate_2_3 = APSK(_apsk_16_labels, [1, 3.15], [np.pi/4, np.pi/12])
APSK_16_rate_3_4 = APSK(_apsk_16_labels, [1, 2.85], [np.pi/4, np.pi/12])
APSK_16_rate_4_5 = APSK(_apsk_16_labels, [1, 2.75], [np.pi/4, np.pi/12])
APSK_16_rate_5_6 = APSK(_apsk_16_labels, [1, 2.70], [np.pi/4, np.pi/12])
APSK_16_rate_8_9 = APSK(_apsk_16_labels, [1, 2.60], [np.pi/4, np.pi/12])
APSK_16_rate_9_10 = APSK(_apsk_16_labels, [1, 2.57], [np.pi/4, np.pi/12])

_apsk_32_labels =  list(map(lambda x: list(map(lambda y: int(y, 2), x)), [['10001', '10101', '10111', '10011'], ['10000', '00000', '00001', '00101', '00100', '10100', '10110', '00110', '00111', '00011', '00010', '10010'], ['11000', '01000', '11001', '01001', '01101', '11101', '01100', '11100', '11110', '01110', '11111', '01111', '01011', '11011', '01010', '11010']]))
APSK_32_rate_3_4 = APSK(_apsk_32_labels, [1, 2.84, 5.27], [np.pi/4, np.pi/12, 0])
APSK_32_rate_4_5 = APSK(_apsk_32_labels, [1, 2.72, 4.87], [np.pi/4, np.pi/12, 0])
APSK_32_rate_5_6 = APSK(_apsk_32_labels, [1, 2.64, 4.64], [np.pi/4, np.pi/12, 0])
APSK_32_rate_8_9 = APSK(_apsk_32_labels, [1, 2.54, 4.33], [np.pi/4, np.pi/12, 0])
APSK_32_rate_9_10 = APSK(_apsk_32_labels, [1, 2.53, 4.30], [np.pi/4, np.pi/12, 0])
