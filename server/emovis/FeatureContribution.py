import numpy as np
from server.ccpca.ccpca.ccpca import CCPCA
#from ccpca import CCPCA


def simplify(feat_mat, emotion_names, feature_names, nFeatures=15):
    print('max-val', feat_mat.max())
    print('min-val', feat_mat.min())

    nFeat, nEmos = feat_mat.shape
    idFeat = []
    for i in range(nEmos):
        tmp = np.abs(feat_mat[:, i])
        tmp = np.argpartition(tmp, -nFeatures)[-nFeatures:]
        idFeat = idFeat + tmp.tolist()
    print(idFeat)
    print(type(idFeat))

    idFeat = np.unique(idFeat)
    feat_mat_final = np.zeros((len(idFeat), nEmos))
    fnames = []
    pos = 0
    print('len: ', len(idFeat))
    for i in idFeat:
        feat_mat_final[pos, :] = feat_mat[i, :]
        fnames.append(feature_names[i])
        pos = pos + 1

    print('max-val', feat_mat_final.max())
    print('min-val', feat_mat_final.min())
    print(feat_mat_final.tolist())
    return np.asmatrix(feat_mat_final).tolist(), emotion_names, fnames


def compute_contribution(features, classes, emotion_names):
    feature_names = features.columns.tolist()
    y = np.array([cls['emotion'] for cls in classes])
    _, n_feats = features.shape
    n_labels = len(emotion_names)
    first_cpc_mat = np.zeros((n_feats, n_labels))
    feat_contrib_mat = np.zeros((n_feats, n_labels))

    # 1. get the scaled feature contributions and first cPC for each label
    ccpca = CCPCA(n_components=1)
    for i, target_label in enumerate(emotion_names):
        target_eq = features[y == i]
        target_diff = features[y != i]

        if len(target_eq) > 0 and len(target_diff) > 0:
            ccpca.fit(
                target_eq,
                target_diff,
                var_thres_ratio=0.5,
                n_alphas=40,
                max_log_alpha=0.5)

            first_cpc_mat[:, i] = ccpca.get_first_component()
            feat_contrib_mat[:, i] = ccpca.get_scaled_feat_contribs()

    print('max-val', feat_contrib_mat.max())
    print('min-val', feat_contrib_mat.min())

    return np.asmatrix(feat_contrib_mat).tolist(), emotion_names, feature_names
