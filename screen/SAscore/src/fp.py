from bitstring import BitArray

import subprocess32
import shlex


def getHexFp(smi_fn):
    """get the finger print in hex
    Keyword Arguments:
    smi_fn -- input file for the molecule in smi format
    """
    cmd = "babel -ismi {} -ofpt - -xN1024".format(smi_fn)
    cmds = shlex.split(cmd)
    stdout = subprocess32.check_output(cmds)
    hex_strs = []
    for line in stdout.splitlines():
        if not line.startswith('>'):
            hex_strs.extend(line.split())
    return hex_strs


def convert2Bits(hex_strs):
    def foo(hex_str):
        c = BitArray(hex=hex_str)
        return c.bin
    return "".join(map(foo, hex_strs))
