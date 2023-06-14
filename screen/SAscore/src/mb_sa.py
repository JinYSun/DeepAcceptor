#!/work/jaydy/anaconda2/envs/my-rdkit-env/bin/python

from rdkit import Chem
from fp import getHexFp, convert2Bits

import luigi
import tempfile
import os

import sascorer


class Path(luigi.Task):

    task_name = luigi.Parameter()

    def getDataDir(self):
        return "/work/jaydy/dat/mb_sa"

    def getSmiPath(self):
        path = os.path.join(self.getDataDir(), self.task_name)
        return path

    def getTopWorkDir(self):
        return "/work/jaydy/working/mb_sa"

    def getWorkDir(self):
        dirname = self.task_name.split('.')[0]
        path = os.path.join(self.getTopWorkDir(), dirname)
        try:
            os.makedirs(path)
        except:
            pass
        return path


class CalculateSAscore(Path):
    def output(self):
        path = os.path.join(self.getWorkDir(), self.task_name + ".sa.txt")
        return luigi.LocalTarget(path)

    def run(self):
        def runSA(ifn, ofn, error_ofn):
            err_ofs = open(error_ofn, 'w')
            with open(ofn, 'w') as ofs:
                with open(ifn, 'r') as ifs:
                    lines = ifs.readlines()
                    print(lines)
                    for line in lines:
                        try:
                            smi = line.split()[0]
                            m = Chem.MolFromSmiles(smi)
                            sa = sascorer.calculateScore(m)
                            ofs.write("%s\t%f\n" % (line.rstrip(), sa))
                        except Exception as inst:
                            err_ofs.write("%s\t%s\n" % (smi, inst.args))
            err_ofs.close()

        ifn = Path(self.task_name).getSmiPath()
        ofn = self.output().path
        error_ofn = ofn + ".err"
        runSA(ifn, ofn, error_ofn)


class Smi2FP(CalculateSAscore):
    def output(self):
        path = os.path.join(self.getWorkDir(), self.task_name + ".sa.fp.txt")
        return luigi.LocalTarget(path)

    def run(self):
        ifn = CalculateSAscore(self.task_name).output().path
        to_write = []
        try:
            tmp_fn = tempfile.mkstemp(suffix='.smi')[1]
            for line in file(ifn):
                smi, name, sa = line.split()
                with open(tmp_fn, 'w') as ofs:
                    ofs.write(smi)
                bits = convert2Bits(getHexFp(tmp_fn))
                to_write.append("{} {} {}\n".format(bits, name, sa))
        except Exception as d:
            print(d)
        finally:
            os.remove(tmp_fn)

        with open(self.output().path, 'w') as ofs:
            ofs.writelines(to_write)


def main(task_name):
    luigi.build(
        [
            CalculateSAscore(task_name), Smi2FP(task_name)
        ], local_scheduler=True)


if __name__ == '__main__':
    import sys
    main(r"J:\screenacc\exe.smi")
