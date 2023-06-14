#!/work/jaydy/anaconda2/envs/my-rdkit-env/bin/python

import luigi
import os
import glob
import subprocess32
import sascorer
from rdkit import Chem


class Path(luigi.Task):
    task_name = luigi.Parameter()

    def getRootWorkDir(self):
        return "/work/jaydy/working/edud_sa"

    def getDataDir(self):
        return "/work/jaydy/dat/tars"

    def getTarPath(self):
        return os.path.join(self.getDataDir(),
                            self.task_name + '.tar.gz')

    def getWorkDir(self):
        path = os.path.join(self.getRootWorkDir(), self.task_name)
        try:
            os.makedirs(path)
        except:
            pass
        return path

    def getUntarredPath(self):
        path = os.path.join(self.getRootWorkDir(),
                            self.task_name)
        return path


class UnTar(Path):
    def requires(self):
        pass

    def output(self):
        path = self.getUntarredPath()
        return luigi.LocalTarget(path)

    def run(self):
        cmds = ['tar', 'xf', self.getTarPath(),
                '-C', self.getRootWorkDir()]
        subprocess32.call(cmds)


class CalculateSAscore(UnTar):

    def getAllSmiFiles(self):
        regx = os.path.join(self.getUntarredPath(),
                            "*/*.sdf-all.smi")
        files = glob.glob(regx)
        return files


class SAscore4Smi(luigi.Task):

    smi_path = luigi.Parameter()

    def requires(self):
        my_dirname = os.path.split(self.smi_path)[0]
        task_name = my_dirname.split('/')[-2]
        return UnTar(task_name)

    def output(self):
        pass

    def runSA(self, ifn, ofn, error_ofn, restart=True, startline=0):
        import sys
        sys.stderr = open('/dev/null', 'w')  # silence the warnings

        if restart is True:
            err_ofs = open(error_ofn, 'w')
            with open(ofn, 'w') as ofs:
                with open(ifn, 'r') as ifs:
                    smis = [_.rstrip() for _ in ifs]
                    for smi in smis:
                        try:
                            m = Chem.MolFromSmiles(smi)
                            sa = sascorer.calculateScore(m)
                            ofs.write("%s\t%f\n" % (smi, sa))
                        except Exception as inst:
                            err_ofs.write("%s\t%s\n" % (smi, inst.args))
            err_ofs.close()
        else:
            err_ofs = open(error_ofn, 'a')
            with open(ofn, 'a') as ofs:
                with open(ifn, 'r') as ifs:
                    smis = [_.rstrip() for _ in ifs]
                    for smi in smis[startline:]:
                        try:
                            m = Chem.MolFromSmiles(smi)
                            sa = sascorer.calculateScore(m)
                            ofs.write("%s\t%f\n" % (smi, sa))
                        except Exception as inst:
                            err_ofs.write("%s\t%s\n" % (smi, inst.args))
            err_ofs.close()

    def run(self):
        out_fn = self.smi_path + '.sa.txt'
        error_ofn = self.smi_path + '.sa.error'

        if not os.path.exists(out_fn):
            print("start %s" % out_fn)
            self.runSA(self.smi_path, out_fn, error_ofn, restart=True)
        else:
            if os.path.exists(error_ofn):
                total_lines = len(open(self.smi_path, 'r').readlines())
                out_lines = len(open(out_fn, 'r').readlines())
                error_lines = len(open(error_ofn, 'r').readlines())
                finished_lines = out_lines + error_lines
                if finished_lines != total_lines:
                    print("restart %s ant continue at %d" % (self.smi_path,
                                                             finished_lines))
                    self.runSA(self.smi_path, out_fn, error_ofn,
                               restart=False,
                               startline=finished_lines)


def main(smi_path):
    luigi.build([
        SAscore4Smi(smi_path),
    ],
                local_scheduler=True)
    pass


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
