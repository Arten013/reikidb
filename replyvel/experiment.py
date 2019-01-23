from .new_policy_match import PolicyMatchObject, PolicyMatchUnit
import shutil
from . import new_policy_match as pm
import types
from abc import ABC, abstractmethod
from . import dataset, ml_models, tokenizer, model_core, alternately_algorithm
from datetime import datetime
import pickle
from itertools import combinations, product
from jstatutree.etypes import code2jname, Article
from .utils.edge import Edge
import csv
from jstatutree.element import Element
from jstatutree.lawdata import ElementNumber
import os
import re
import pandas as pd
import yaml
from pathlib import Path
from typing import Callable, Mapping, Union, Generator, NewType, Tuple, List, Set, Dict, Any, Type, TypeVar
import numpy as np
import subprocess
import simstring

Pathlike = NewType('Pathlike', Union[Path, str])

X = TypeVar('X')


class DirectoryStructure(object):
    def __init__(self, base: Pathlike, *, mapping_file: Pathlike = None, mapping: Mapping = None, **spec_strs):
        self.base = Path(base)
        self.spec_strs = spec_strs
        if mapping is not None and mapping_file is None:
            self.structure = mapping
        elif mapping_file is not None and mapping is None:
            self.structure = self.load(mapping_file)
        else:
            raise ValueError(
                'Invalid arguments: {0} as mapping_file and {1} as mapping'.format(repr(mapping_file), repr(mapping)))
        self.flatten_structure: Dict[str, Dict] = {'file': {}, 'dir': {}}
        for type_, key, path in self.bfs_walk():
            self.flatten_structure[type_][key] = path

    @staticmethod
    def load(path: Pathlike) -> Mapping:
        # todo: load from both json and yaml file.
        raise Exception('Not implemented')

    def get_path(self, key: str, **kwargs) -> Path:
        # find directory specified by key.
        # key includes some kwargs for complete files
        path = self.flatten_structure['file'].get(key, None) or self.flatten_structure['dir'].get(key, None)
        if path is None:
            return Path(os.devnull)
        return Path(str(path).format(**kwargs))

    def check(self, *keys):
        # Check the structure if it satisfy all the components.
        # It also check if the file structure is valid.
        components = set()
        paths = set()
        for type_, key, path in self.bfs_walk():
            if key in components:
                raise Exception('key duplication: {}'.format(key))
            if path in paths:
                raise Exception('path duplication: {}'.format(path))
            paths.add(path)
            components.add(key)
        if not set(keys).issubset(components):
            raise Exception('components shortage:\n{}'.format(set(keys) - components))

    def decode_tag(self, tag: str) -> Tuple[str, str]:
        if ':' not in tag:
            return re.split('\.', tag)[0], tag
        ret = re.split(':', tag)
        assert len(ret) == 2, 'Invalid tag: {}'.format(tag)
        if ret[1][0] == '#':
            ret[1] = self.spec_strs[ret[1][1:]]
        return ret[0], ret[1]

    def bfs_walk(self) -> Generator[Tuple[str, str, Path], None, None]:
        queue = [(self.base, k, v) for k, v in self.structure.items()]
        tail = 0
        while len(queue) > tail:
            prefix, key, values = queue[tail]
            tail += 1
            if key[0] == '#':
                if key == '#files':
                    for tag in values:
                        fkey, fname = self.decode_tag(tag)
                        yield ('file', fkey, prefix / fname)
                else:
                    raise ValueError('Invalid Meta-Tag {}'.format(key))
            else:
                fkey, fname = self.decode_tag(key)
                yield ('dir', fkey, prefix / fname)
                queue.extend([(prefix / fname, k, v) for k, v in values.items()])

    _create_dir_ptn = re.compile('{[^/{}]+}')

    def create_dirs(self, exist_ok=False):
        # create all directories in the structure.
        for key, path in self.flatten_structure['dir'].items():
            if self._create_dir_ptn.search(str(path)):
                continue
            os.makedirs(str(path), exist_ok=True)


class ReikiAnnotation(object):
    def __init__(self, path: Pathlike):
        self.labels = []
        self.label_tags = None
        self.annotations: Dict[str, Dict[str, Set[ElementNumber]]] = {}
        self.load(path)

    _ordcode_ptn = re.compile('\d{2}-\d{6}-\d{4}$')

    def load(self, src: Pathlike):
        # load file
        stack = [Path(src)]
        while len(stack) > 0:
            path = stack.pop()
            if path.is_dir():
                stack.extend(path.iterdir())
                continue
            if path.suffix not in ['.yaml', '.yml']:
                continue
            if self._ordcode_ptn.match(path.stem):
                with path.open() as fd:
                    self.annotations[re.sub('-', '/', path.stem)] = {
                        k: set([ElementNumber(n) for n in nums]) for k, nums in yaml.load(fd).items()
                    }
            elif re.match(path.stem, 'labels'):
                with path.open() as fd:
                    self.label_tags = yaml.load(fd)
                    self.labels = list(self.label_tags.keys())
        assert self.label_tags is not None, 'Failed to load labels.yaml'

    def node_to_labels(self, node: Element) -> Set[str]:
        # transform node (Element obj) to labels according to annotation
        rcode = node.code[:14]
        labels = set()
        articles = list(node.iter('Article'))
        for label, nums in self.annotations[rcode].items():
            for a in articles:
                if a.num in nums:
                    labels.add(label)
                    break
        return labels

    def check(self, qcode: str, tcode: str, output: Dict) -> Tuple[int, int, int]:
        # check whether tnode's labels are correct according to annotation
        # return TP, TN, FP
        entire_tp, entire_tn, entire_fp = 0, 0, 0
        for label in self.labels + ['others']:
            query_nums = self.annotations[qcode].get(label, set())
            true_nums = self.annotations[tcode].get(label, set())
            ans_nums = output.get(label, set())
            if label == 'others' or len(query_nums) == 0:
                entire_fp += len(ans_nums)
                continue
            tp = len(true_nums & ans_nums)
            entire_tp += tp
            entire_tn += len(true_nums) - tp
            entire_fp += len(ans_nums) - tp
        return entire_tp, entire_tn, entire_fp


# Analyze PolicyMatch obj and transform to human recognizable results
class MatchAnalyzer(object):
    class MatchCSVBuilder(object):
        def __init__(self, path, header: List = None):
            self.path = path
            self.fd = open(self.path, 'w', newline='')
            self.writer = csv.writer(self.fd)
            self.writer.writerow(header)
            self.writer.writerow(['query', '', '', 'target', '', '', ''])
            self.writer.writerow(["part", "labels", "captions", "part", "labels", "captions", "score"])

        def close(self):
            self.fd.close()
            del self.fd
            del self.writer

        def write_edge(self, edge: Edge, qlabels, tlabels):
            # write edge
            self.writer.writerow([
                code2jname(edge.qnode.code),
                '/'.join(list(qlabels)),
                '/'.join([e.caption for e in edge.qnode.iter('Article') if len(e.caption)]),
                code2jname(edge.tnode.code),
                '/'.join(list(tlabels)),
                '/'.join([e.caption for e in edge.tnode.iter('Article') if len(e.caption)]),
                edge.score
            ])

    DEFAULT_STRUCTURE = {
        'main:#timestamp':
            {
                '#files': [
                    'analyzer:analyzer.pkl',
                    'report.txt',
                    'description.txt',
                    'params.csv',
                    'result.csv',
                ],
                'annotation': {},
                'graphs': {
                    'graph_gov_dir:{ord_code}': {}
                },
                'match_ords': {
                    '#files': [
                        'match_ord:{ord_code}.csv',
                        'match_ord_arti:{ord_code}_arti.csv',
                    ]
                }
            }
    }
    COMPONENT_KEYS = ['main', 'analyzer', 'annotation']

    def __init__(self, match: PolicyMatchObject, *, root_path: Pathlike = None, structure: DirectoryStructure = None):
        self.match = match
        self.timestamp = str(datetime.now().strftime('%y%m%d-%H%M%S'))
        if root_path is None and structure is not None:
            self.structure = structure
        elif root_path is not None and structure is None:
            self.structure = DirectoryStructure(root_path, mapping=self.DEFAULT_STRUCTURE, timestamp=self.timestamp)
        else:
            raise Exception('Invalid Argument (You must pass either root_path or structure_obj)')
        self.structure.check(*self.COMPONENT_KEYS)
        self.structure.create_dirs(exist_ok=False)

    def save(self):
        # todo : pickle self
        pickle.dump(file=self.structure.get_path('analyzer'), obj=self)

    def write_description(self, content: str):
        path = self.structure.get_path('description')
        f = path.open('w')
        f.write(content)
        f.close()

    def add_report(self, content: str):
        path = self.structure.get_path('report')
        if path.exists():
            f = path.open('w')
        else:
            f = path.open('a')
        f.writelines([str(datetime.now()) + '\n', content.rstrip() + '\n\n'])
        f.close()

    def set_annotation(self, path: Pathlike):
        import shutil
        tar_path = self.structure.get_path('annotation')
        shutil.rmtree(str(tar_path))
        shutil.copytree(str(path), str(tar_path))

    @classmethod
    def load(cls, base: Pathlike):
        return pickle.load(base)

    def evaluate(self, render_graphs: bool = False, **render_args) -> pd.DataFrame:
        # calculate precision, recall, F-val, and other results.
        # generate detail results and save as logs
        df = pd.DataFrame(columns=['TP', 'TN', 'FP', 'Precision', 'Recall'])
        annotation = ReikiAnnotation(self.structure.get_path('annotation'))
        macro = {}
        for tkey, u in self.match.units.items():
            if tkey not in annotation.annotations:
                continue
            model_output: Dict[str, Set[str]] = {l: set() for l in annotation.labels + ['others']}
            macro[tkey] = {}

            header = "{0} vs. {1}".format(
                self.match.units[tkey].query_tree.lawdata.name,
                self.match.units[tkey].match_tree.lawdata.name,
            )
            match_csv_path = self.structure.get_path('match_ord', ord_code=re.sub('/', '-', tkey))
            match_csv = MatchAnalyzer.MatchCSVBuilder(match_csv_path, header=[header])
            match_csv_arti_path = self.structure.get_path('match_ord_arti', ord_code=re.sub('/', '-', tkey))
            match_arti_csv = MatchAnalyzer.MatchCSVBuilder(match_csv_arti_path, header=[header])

            if render_graphs:
                self.render_comp_table(**render_args)
            for skey, edges, edge_store in u.find_matching_edge(threshold=0.0):
                assert skey == 'default', 'this class do not applicable for multi-score.'
                for edge in edges:
                    qlabels = annotation.node_to_labels(edge.qnode)
                    tlabels = annotation.node_to_labels(edge.tnode)
                    match_csv.write_edge(edge, qlabels, tlabels)
                    if 'Item' in edge.qnode.code + edge.tnode.code:
                        continue
                    if 'Paragraph' in edge.qnode.code + edge.tnode.code:
                        continue
                    for qarti in edge.qnode.iter('Article'):
                        for tarti in edge.tnode.iter('Article'):
                            article_edge = Edge(qarti, tarti, edge.score)
                            arti_qlabels = annotation.node_to_labels(qarti)
                            arti_tlabels = annotation.node_to_labels(tarti)
                            if len(arti_qlabels) == 0 and len(arti_tlabels) == 0:
                                continue
                            match_arti_csv.write_edge(article_edge, arti_qlabels, arti_tlabels)
                            if len(arti_qlabels) == 0:
                                arti_qlabels.add('others')
                            for ql in arti_qlabels:
                                model_output[ql].add(tarti.num)

            #print(model_output)
            match_csv.close()
            match_arti_csv.close()
            tp, tn, fp = annotation.check(str(list(self.match.units.values())[0].query_tree.lawdata.code), tkey,
                                          model_output)
            df.loc[tkey] = [tp, tn, fp, tp / (tp + fp) if (tp + fp) > 0 else 0, tp / (tp + tn) if (tp + tn) > 0 else 0]
        micro = df.aggregate(['sum'])
        micro.loc['sum', 'Precision'] = micro.loc['sum', 'TP'] / (micro.loc['sum', 'TP'] + micro.loc['sum', 'FP'])
        micro.loc['sum', 'Recall'] = micro.loc['sum', 'TP'] / (micro.loc['sum', 'TP'] + micro.loc['sum', 'TN'])
        df = df.append(df.aggregate(['mean']).rename({'mean': 'macro'}))
        df = df.append(micro.rename({'sum': 'micro'}))
        df.to_csv(self.structure.get_path('result'))
        return df

    @staticmethod
    def render_core(path: Pathlike, format: str) -> bool:
        render_options = ['-x', '-Goverlap=scale']
        for j in range(len(render_options) + 1):
            for options in combinations(render_options, j):
                render = subprocess.Popen(
                    ['dot'] + list(options) + ['-T{}'.format(format), '-O', str(path.resolve())],
                    stderr=subprocess.PIPE
                )
                out = render.stderr.read()
                if len(out) == 0:
                    return True
                print('err:', out.decode())
            print('retry')
        return False

    def render_comp_table(self, unit: PolicyMatchUnit, output_format='pdf'):
        tkey = unit.match_tree.lawdata.code
        path = self.structure.get_path('graph_gov_dir', ord_code=re.sub('/', '-', tkey))
        os.makedirs(str(path), exist_ok=True)
        name = 'main'
        G = unit.comp_table(threshold=0, sub_branch='ALL')
        with (path / (name + '.pdf')).open('w') as f:
            f.write(G.source)
        if self.render_core(path / name, output_format):
            print("render:", path, unit.match_tree.lawdata.name)
        else:
            print('skip: ', str(path / name))
        for i, (t, e, G) in enumerate(
                unit.unit_comp_tables
                    (threshold=0, sub_branch='ALL', match_root='{}/Law(1)'.format(tkey),
                     query_root='{}/Law(1)'.format(unit.query_tree.lawdata.code))
        ):
            if 'Item' in str(e) or 'Paragraph' in str(e):
                continue
            e.score = round(e.score, 3)
            name = "{0}-{1}".format(str(e.qnode), str(e.tnode))
            G = unit.comp_table(threshold=0, sub_branch='ALL')
            with (path / (name + '.pdf')).open('w') as f:
                f.write(G.source)
                if self.render_core(path / name, output_format):
                    print("render:", path, unit.match_tree.lawdata.name)
                else:
                    print('skip: ', str(path / name))


class Experiment(object):
    @staticmethod
    def open_setting(setting_path: Pathlike):
        # fuction to load setting file and create instance of setting
        setting_obj = Experiment.Setting()
        with open(setting_path) as f:
            setting: Mapping = yaml.load(f)
        for key, val in setting.get('var', {}).items():
            setting_obj.var.add(key, val)
        setting_obj._current_var.update(setting_obj.var.consts)
        for key, fname in setting.get('import', {}).items():
            setting_obj.sub_settings[key] = Experiment.open_setting(Path(setting_path).parent / fname)
            setting_obj.sub_settings[key]._current_var = setting_obj._current_var
        setting_obj.content = setting.get('content', setting)
        return setting_obj

    class Setting(object):
        class Var(object):
            def __init__(self):
                self.var_tags = []
                self.callables = {}
                self.lists = {}
                self.consts = {}

            def add(self, tag: str, value: Any):
                if isinstance(value, list):
                    self.var_tags.append(tag)
                    self.lists[tag] = value
                    return
                if re.match('#eval:', value):
                    value = eval(value[6:], globals(), locals())
                if tag in self.var_tags:
                    raise ValueError('Variable tag duplication: {}'.format(tag))
                self.var_tags.append(tag)
                if isinstance(value, types.GeneratorType):
                    self.lists[tag] = list(value)
                elif isinstance(value, types.FunctionType):
                    self.callables[tag] = value
                elif isinstance(value, np.ndarray):
                    self.lists[tag] = value.tolist()
                elif isinstance(value, list):
                    self.lists[tag] = value
                else:
                    self.consts[tag] = value

            def grid_iterator(self):
                list_tags = list(self.lists.keys())
                combinations = list(product(*[self.lists[k] for k in list_tags]))
                print('grid iteration candidate: {}'.format(len(combinations)))
                for i, list_vals in enumerate(combinations):
                    vals = {t: v for t, v in zip(list_tags, list_vals)}
                    vals.update(self.consts)
                    tasks = list(self.callables.keys())
                    allowed_failure_count = (len(tasks) ^ 2) // 2 + 1
                    while len(tasks) > 0:
                        callable_tag = tasks.pop()
                        try:
                            vals[callable_tag] = self.callables[callable_tag](vals)
                        except Exception as e:
                            tasks.append(callable_tag)
                            if allowed_failure_count > 0:
                                allowed_failure_count -= 1
                            else:
                                raise ValueError('Cannot resolve callable relation')
                    print(vals)
                    yield vals
                    print('task {0}/{1} finished'.format(i+1, len(combinations)))
                return

        def __init__(self):
            self.sub_settings: Dict[str, Experiment.Setting] = {}
            self.var = self.__class__.Var()
            self._current_var: Dict = {}
            self.content: Mapping = {}
            self._var_gen = self.var.grid_iterator()

        def __getitem__(self, key: str):
            item = self.content[key]
            if isinstance(item, (types.MappingProxyType, dict)):
                setting_obj = self.__class__()
                setting_obj.sub_settings = self.sub_settings
                setting_obj.var = self.var
                setting_obj._current_var = self._current_var
                assert id(setting_obj._current_var) == id(self._current_var)
                setting_obj.content = item
                return setting_obj
            elif isinstance(item, list):
                return item
            elif item.startswith('$'):
                ret = self._current_var[item[1:]]
                return ret
            elif '->' in item:
                sub_setting_tag, path = item.split('->', maxsplit=1)
                return self.sub_settings[sub_setting_tag].retrieve(Path(path))
            return item

        def decode_var(self, var):
            if len(var) <= 1:
                return var
            ret = self._current_var.get(var[1:], var)
            return ret

        def get(self, key: str, default: X = None):
            try:
                return self[key]
            except:
                return default

        def retrieve(self, path: Pathlike):
            path = Path(path)
            parts = path.parts
            if len(parts) == 0:
                return self
            if parts[0] == '/':
                if len(path.parts) == 1:
                    return self
                if len(path.parts) == 2:
                    return self[path.parts[1]]
                return self[path.parts[1]].retrieve(Path(*path.parts[2:]))
            else:
                if len(path.parts) == 1:
                    return self[path.parts[0]]
                return self[path.parts[0]].retrieve(Path(*path.parts[1:]))

        def next_vars(self):
            # switch self.var to next vars
            # return bool (was able to switch or not)
            item = next(self._var_gen, None)
            if item is None:
                self._var_gen = self.var.grid_iterator()
                self._current_var.clear()
                self._current_var.update(self.var.consts)
                return False
            self._current_var.clear()
            self._current_var.update(item)
            return True

        def reconstruct_df(self, df: pd.DataFrame, grid_num):
            df.loc[:, 'grid_num'] = grid_num
            for k, v in self._current_var.items():
                df.loc[:, k] = v
            df.index = df.index.set_names(['reiki'])
            df.reset_index(inplace=True)

        def to_dict(self) -> Dict:
            return {
                k: v.to_dict() if isinstance(v, self.__class__) else v
                for k in self.content.keys()
                for v in [self[k]]
            }

        def items(self):
            yield from self.to_dict().items()

    class BuilderAbstract(ABC):
        SETTING_ROOT = '/'

        def __init__(self):
            self.yaml_path: Path = Path('')
            self.setting: Experiment.Setting = {}

        def set_setting(self, setting):
            self.setting = setting.retrieve(self.SETTING_ROOT)

        @abstractmethod
        def build(self):
            pass

    class TestsetBuilder(BuilderAbstract):
        SETTING_ROOT = '/testset'

        class TestSet(object):
            def __init__(self, query_code: str, target_codes: List[str]):
                self.query_code = query_code
                self.target_codes = target_codes

        def build(self):
            query_code: str = self.setting['query']
            target_codes: List[str] = self.setting['retrieval_space']
            return self.TestSet(query_code, target_codes)

    class LeafMatchBuilder(BuilderAbstract):
        SETTING_ROOT = '/leaf_matching'

        def build(self):
            model_cls: Type[model_core.JstatutreeModelCore] = getattr(ml_models, self.setting['model']['type'])
            if model_cls in [ml_models.SimString]:
                db = dataset.JstatutreeDB(path=self.setting['model']['training_set']['path'])
            else:
                db = dataset.TokenizedJstatutreeDB(
                    path=self.setting['model']['training_set']['path'],
                    tokenizer=tokenizer.MecabTokenizer("mecab_ipadic")
                )
            model = model_cls(
                db=db,
                tag=self.setting['model']['tag'],
                unit='XSentence',
                **self.setting['model']['params'].to_dict()
            )
            model.fit()
            return model

    class ScorerBuilder(BuilderAbstract):
        SETTING_ROOT = '/retrieve/model/scorer'

        def set_algorithm(self, algorithm):
            self.algorithm = algorithm

        def build(self):
            scorer_name = self.setting['main']['tag']
            built_scorers = {}
            for tier, scorers in sorted(self.setting['sub'].items(), key=lambda x: int(x[0][4:])):
                for name, scorer_setting in scorers.items():
                    scorer_cls = getattr(getattr(self.algorithm, 'Scorer'), scorer_setting['cls'])
                    scorer = scorer_cls(**scorer_setting.get('params', {}))
                    for sub_scorer in scorer_setting.get('sub_scorers', []):
                        scorer.add(built_scorers[sub_scorer['tag']], **{k: self.setting.decode_var(v) for k, v in sub_scorer.get('params', {}).items()})
                    if scorer_name == name:
                        print("build main scorer:", scorer)
                        return scorer
                    built_scorers[name] = scorer

    DEFAULT_STRUCTURE = {
        'main:#timestamp':
            {
                '#files': [
                    'analyzer:analyzer.pkl',
                    'report.txt',
                    'description.txt',
                    'params.csv',
                    'aggregated_result.csv',
                ],
                'annotation': {},
                'results':{
                    'result_dir:#grid_id': {
                        '#files': [
                            'result.csv',
                        ],
                        'match_ords': {
                            '#files': [
                                'match_ord:{ord_code}.csv',
                                'match_ord_arti:{ord_code}_arti.csv',
                            ]
                        }
                    }
                }
            }
    }

    def __init__(self, setting_path: Pathlike, result_path: Pathlike):
        self.leaf_match_model: ml_models.JstatutreeModelCore = None
        self.result_path = result_path
        self.timestamp = str(datetime.now().strftime('%y%m%d-%H%M%S'))
        self.setting = Experiment.open_setting(setting_path)
        self.algorithm = globals()[self.setting.retrieve('/retrieve/algorithm')]

        self.leaf_match_model_builder = Experiment.LeafMatchBuilder()
        self.leaf_match_model_builder.set_setting(self.setting)

        self.testset_builder = Experiment.TestsetBuilder()
        self.testset_builder.set_setting(self.setting)

        self.scorer_builder = Experiment.ScorerBuilder()
        self.scorer_builder.set_setting(self.setting)
        self.scorer_builder.set_algorithm(self.algorithm)

        self.traverser_cls = getattr(self.algorithm, self.setting['retrieve']['model']['traverser'])

        self.activator = getattr(pm.SimilarityActivator, self.setting['retrieve']['model']['activator'])

    def run(self):
        leaf_match_model = self.leaf_match_model_builder.build()
        leaf_match_model.restrict_rspace(self.setting.retrieve('/testset/retrieval_space'))
        grid_num = 0
        all_df = pd.DataFrame()
        tar_path = self.get_structure(grid_num).get_path('annotation')
        shutil.copytree(str("/home/jovyan/develop/annotation"), str(tar_path))
        annotated_codes = list(ReikiAnnotation("/home/jovyan/develop/annotation").annotations.keys())
        leaf_match_cache = {}
        while self.setting.next_vars():
            finger_print = self.setting._current_var["theta_s"]
            match_factory = leaf_match_cache.get(finger_print, None)
            if match_factory is None:
                match_factory = leaf_match_model.build_match_factory(
                    query_key=self.setting.retrieve('/testset/query'),
                    theta=self.setting.retrieve('/leaf_matching/theta_s'),
                    match_factory_cls=pm.PolicyMatchFactory,
                )
                leaf_match_cache[finger_print] = match_factory
            match = match_factory.matching(
                scorer=self.scorer_builder.build(),
                traverser_cls=self.traverser_cls,
                activator=self.activator(),
                threshold=self.setting.retrieve('/retrieve/model/theta_v'),
                target_codes=annotated_codes
            )
            analyzer = MatchAnalyzer(match, structure=self.get_structure(grid_num))
            df = analyzer.evaluate()
            self.setting.reconstruct_df(df, grid_num)
            all_df = pd.concat([all_df, df])
            print(df)
            grid_num += 1
        all_df = all_df.set_index(['grid_num', 'reiki'])
        all_df.to_csv(self.get_structure(0).get_path('aggregated_result'))
        return all_df

    def get_structure(self, i: int) -> DirectoryStructure:
        return DirectoryStructure(
            self.result_path,
            mapping=self.DEFAULT_STRUCTURE,
            timestamp=self.timestamp,
            grid_id='grid-{0:04}'.format(i))