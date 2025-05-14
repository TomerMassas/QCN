import os
import pandas as pd
import shutil


if __name__ == '__main__':
    video_name = 'Film'
    QCN_csv_result_folder = fr'C:\Users\TomerMassas\Documents\GitHub\QCN\results\KonIQ10K\CTV29\{video_name}'
    csv_paths = os.listdir(QCN_csv_result_folder)
    csv_paths = [os.path.join(QCN_csv_result_folder, path) for path in csv_paths if path.endswith('.csv')]

    result_dir = fr'C:\Users\TomerMassas\Desktop\Video project\video scene detection\tests\{video_name}\best_frame_per_segment_QCN'
    os.makedirs(result_dir, exist_ok=True)
    for csv_p in csv_paths:
        df = pd.read_csv(csv_p)
        best_frame = df.sort_values(by='predicted_score', ascending=False).head(1)['image_name'].values[0]
        segment_name = os.path.basename(csv_p).split('_')[-1].split('.')[0]
        shutil.copy(fr'C:\Users\TomerMassas\Desktop\Video project\video scene detection\tests\{video_name}\frames\frames of segments\segment_{segment_name}\{best_frame}',
                     os.path.join(result_dir, f'segment_{segment_name}_{best_frame}'))


