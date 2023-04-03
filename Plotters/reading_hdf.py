import h5py
import numpy as np
from plotter_nparray import plot

def sample_cloud(vertices_c, faces_vc, size=2**10, return_eval_cloud=False):
    polygons = vertices_c[faces_vc]
    cross = np.cross(polygons[:, 2] - polygons[:, 0], polygons[:, 2] - polygons[:, 1])
    areas = np.sqrt((cross**2).sum(1)) / 2.0

    probs = areas / areas.sum()
    p_sample = np.random.choice(np.arange(polygons.shape[0]), size=2 * size if return_eval_cloud else size, p=probs)

    sampled_polygons = polygons[p_sample]

    s1 = np.random.random((2 * size if return_eval_cloud else size, 1)).astype(np.float32)
    s2 = np.random.random((2 * size if return_eval_cloud else size, 1)).astype(np.float32)
    cond = (s1 + s2) > 1.
    s1[cond] = 1. - s1[cond]
    s2[cond] = 1. - s2[cond]

    sample = {
        'cloud': (sampled_polygons[:, 0] +
                  s1 * (sampled_polygons[:, 1] - sampled_polygons[:, 0]) +
                  s2 * (sampled_polygons[:, 2] - sampled_polygons[:, 0])).astype(np.float32)
    }

    if return_eval_cloud:
        sample['eval_cloud'] = sample['cloud'][1::2].copy().T
        sample['cloud'] = sample['cloud'][::2].T
    else:
        sample['cloud'] = sample['cloud'].T

    #plot(sample['cloud'])

    return sample

with h5py.File(r'D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\data\ShapeNetCore55v2_meshes_resampled.h5', 'r') as fin:    
    
    vertices_c_bounds = np.empty(fin['train_vertices_c_bounds'].shape, dtype=np.uint64)
    fin['train_vertices_c_bounds'].read_direct(vertices_c_bounds)
    vertices = np.array(fin['train_vertices_c'][vertices_c_bounds[0]:vertices_c_bounds[1]], dtype=np.float32)
    
    faces_bounds = np.empty(fin['train_faces_bounds'].shape, dtype=np.uint64)
    fin['train_faces_bounds'].read_direct(faces_bounds)
    faces = np.array(fin['train_faces_vc'][faces_bounds[0]:faces_bounds[1]],dtype=np.uint32)

#sampled_pts = sample_cloud(vertices, faces)['cloud']

#plot(sampled_pts)
print(sample_cloud(vertices, faces).keys())