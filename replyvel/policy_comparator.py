
class Hoge(object):
    def __init__(self)
def tree_cover_dict(query_vectors, qtags, threshold):
    count_dict = {}
    for qtag, td_pairs in zip(qtags, lsi.most_similar(query_vectors, name_as_tag=False, topn=100)):
        for tag, distance in td_pairs:
            if distance > threshold:
                break
            key = '/'.join(Path(tag).parts[:3])
            if key not in count_dict:
                count_dict[key] = [(tag, qtag, distance) ]
            else:#if qtag not in [i[1]for i in count_dict[key]]:
                count_dict[key].append( (tag, qtag, distance) )
    return OrderedDict(sorted(count_dict.items(), key=lambda t: -len(t[1])))

def calc_coverage(qmcode, tmcode, threshold, lsi):
    qlsi = lsi.get_submodel(qmcode)
    tlsi = lsi.get_submodel(tmcode)
    print(len(qlsi.tagged_vectors), len(tlsi.tagged_vectors))
    ret = {}
    for qcode in list(qlsi.db.jstatutree_db.iterator(include_key=True, include_value=False))[:50]:
        qtags, qvectors = qlsi.keys_to_vectors(qcode, return_keys=True)
        #print(len(qtags))
        tcd = tree_cover_dict(qvectors, qtags, threshold, tlsi)
        if len(tcd) > 0:
            ret[qcode] = (len(tcd[list(tcd.keys())[0]])/len(qtags), tcd)
        else:
            ret[qcode] = (0, tcd)
        print(qcode, qlsi.db.jstatutree_db.get(qcode).lawdata.name)
        print('coverage: {0:.3f}'.format(ret[qcode][0]), '({0}/{1})'.format(len(tcd[list(tcd.keys())[0]]), len(qtags)) if ret[qcode][0] > 0 else '' )
    return ret

def retrieve_subtrees(threshold, leaves, lsi, tcd):
    grouped_leaves = groupby(sorted(leaves, key=lambda x: '/'.join(Path(x).parts[:3])), key=lambda x: '/'.join(Path(x).parts[:3]))
    for lc, gl in grouped_leaves:
        gl = list(gl)
        #print(lc, len(gl))
        law = lsi.db.get_jstatutree(lc)
        yield from retrieve_subelements(law.getroot(), threshold, list(gl), tcd)

def retrieve_subelements(elem, threshold, leaves, tcd):
    gc_subtrees = []
    c_subtrees = []
    if elem.code in leaves:
        for k, v in tcd.items():
            if k not in elem.code:
                continue
            for ttag, qtag, d in v:
                if ttag == elem.code:
                    return [(elem, [(qtag, ttag)])]
    if len(list(elem)) == 0:
        return []
    for c in list(elem):
        for subtree, subtree_qtags in retrieve_subelements(c, threshold, leaves, tcd):
            if c == subtree:
                c_subtrees.append([subtree, subtree_qtags])
            elif c.code not in leaves:
                c_subtrees.append([subtree, subtree_qtags])
    if len(c_subtrees)/len(list(elem)) > threshold:
        #print('unite:', [str(e) for e, qts in c_subtrees], '->', elem.code, len(c_subtrees)/len(list(elem)) )
        return [(elem, [qt for e, qts in c_subtrees for qt in qts])]+gc_subtrees
    else:
        #len(c_subtrees) and print('not unite:', [str(e) for e in c_subtrees], len(c_subtrees)/len(list(elem)) )
        return c_subtrees+gc_subtrees
