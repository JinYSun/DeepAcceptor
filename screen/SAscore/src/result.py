#!/work/jaydy/anaconda2/envs/my-rdkit-env/bin/python

import luigi
import random
import os
from edud import Path as EDUD_PATH
from glob import glob


class Sample(EDUD_PATH):
    def output(self):
        path = os.path.join(self.getWorkDir(),
                            self.task_name + 'sample_sa.txt')
        return luigi.LocalTarget(path)

    def run(self):
        work_dir = self.getWorkDir()
        regx = work_dir + "/*/*.smi.sa.txt"
        fns = glob(regx)
        sa_vals = []
        for fn in fns:
            with open(fn, 'r') as ifs:
                for line in ifs:
                    tokens = line.split()
                    if len(tokens) == 2:
                        sa = float(tokens[-1])
                        sa_vals.append(sa)

        if len(sa_vals) > 1000:
            sa_vals = random.sample(sa_vals, 1000)

        with open(self.output().path, 'w') as ofs:
            for val in sa_vals:
                ofs.write("%f\n" % (val))


def main(task_name="output-aofb"):
    luigi.build([
        Sample(task_name)
    ],
                local_scheduler=True)
    pass

if __name__ == '__main__':
    import sys
    main(task_name="../data/sample.smi")
