import sys
sys.path.append('./')
import os
from utils.module_utils import load_pkl, save_json
from tqdm import tqdm


root = R'data/GigaCrowd'
annot = os.path.join(root, R'annot/test.pkl') 
output = 'output/results/results/predict.json'
params_path = 'output/final/gigacrowd/images/params/images'

output_annot = {}

losses = []
count = 0
annot = load_pkl(annot)
for seq_id, seq in enumerate(annot):
    for frame in tqdm(seq, total=len(seq)):
        _, seq_name, frame_name = frame['img_path'].replace('\\', '/').split('/')

        image_dict = {}
        h, w = frame['h_w']
        f = (h**2 + w**2)**0.5
        tx = w/2.
        ty = h/2.
        intrinsic_matrix = [[f,0,tx],[0,f,ty],[0,0,1]]
        image_dict['Intrinsic_matrix'] = intrinsic_matrix
        # intri = np.array(intrinsic_matrix)

        person_list = []
        for person in frame:
            if person in ['h_w', 'img_path']:
                continue
            name = frame_name.split('.')[0]
            params = os.path.join(params_path, seq_name, name, person + '.pkl')
            params = load_pkl(params)
            person_dict = {}
            person_dict['pose'] = params['pose'].tolist()
            person_dict['shape'] = params['betas'].tolist()
            person_dict['trans_cam'] = params['trans'].tolist()
            person_dict['scale_smpl'] = 1.0
            person_dict['gender'] = 'neutral'
            count += 1

            person_list.append(person_dict)
        
        image_dict['person_list'] = person_list
        output_annot[frame_name] = image_dict


print('Total count: %d' %count)
print('Save to %s' %output)
save_json(output, output_annot)




