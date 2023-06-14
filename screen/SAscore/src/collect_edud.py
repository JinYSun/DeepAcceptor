#!/work/jaydy/anaconda2/envs/my-rdkit-env/bin/python


import luigi
import json
from result import Sample


class CheckProgress(luigi.Task):

    def output(self):
        path = "../data/true_unfinished.txt"
        return luigi.LocalTarget(path)

    def run(self):
        print("path total out error")

        ofs = open(self.output().path, 'w')
        for smi_path in open("../data/unfinished.txt").readlines():
            smi_path = smi_path.rstrip()
            out_fn = smi_path + '.sa.txt'
            error_ofn = smi_path + '.sa.error'
            total_lines = len(open(smi_path, 'r').readlines())
            out_lines = len(open(out_fn, 'r').readlines())
            error_lines = len(open(error_ofn, 'r').readlines())
            finished_lines = out_lines + error_lines
            if finished_lines != total_lines:
                ofs.write("%s %d %d %d\n" % (smi_path,
                                             total_lines,
                                             out_lines,
                                             error_lines))
        ofs.close()


class CollectAllSAscores(luigi.Task):

    def getTaskNames(self):
        return [_.rstrip() for _ in file("../data/edud_jobs.txt")]

    def output(self):
        all_scores_path = "../data/edud_scores.json"
        return luigi.LocalTarget(all_scores_path)

    def run(self):
        data = {}
        for task_name in self.getTaskNames():
            sample_path = Sample(task_name).output().path
            vals = [float(_.rstrip()) for _ in file(sample_path)]
            data[task_name] = vals

        with open(self.output().path, 'w') as ofs:
            ofs.write(json.dumps(data, indent=4, separators=(',', ': ')))


def main():
    luigi.build([
        CollectAllSAscores()
    ],
                local_scheduler=True)
    pass


if __name__ == '__main__':
    main()


