"""
@Author: Conghao Wong
@Date: 2022-06-21 09:38:13
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 14:56:56
@Description: Structures to manage trajectory datasets.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Union

import cv2
import numpy as np

from ..__base import BaseObject
from ..args import BaseArgTable as Args
from ..utils import dir_check
from .__agent import Agent
from .__dataset import Dataset, VideoClip
from .__maps import MapManager
from .__trajectory import Trajectory

INIT_POSITION = 10000
MAP_HALF_SIZE = 50  # Local map's half size

class TrajMapNotFoundError(FileNotFoundError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class VideoClipManager(BaseObject):
    """
    VideoClipManager
    ----------------
    Manage all training data from one video clip.

    Properties
    ----------
    ```python
    >>> self.args   # args
    >>> self.name   # name
    >>> self.dataset # dataset info
    ```

    Public Methods
    --------------
    ```python
    # Sample train data from dataset
    (method) sample_train_data: (self: DatasetManager) -> list[agent_type]

    # Load dataset files
    (method) load_data: (self: DatasetManager) -> Any
    ```
    """

    def __init__(self, args: Args, name: str,
                 custom_list: list[np.ndarray] = []):

        super().__init__()

        self.args = args
        self.name = name

        self.video_clip = VideoClip.get(name)
        self.custom_list = custom_list
        self.agent_count = None
        self.trajectories: list[Trajectory] = None

    def load_csv(self, name) -> tuple[dict[int, np.ndarray], list]:
        """
        Read trajectory data from csv file.

        :param name: name of the dataset. See Details in `datasetManager.py`
        :return persons: data sorted by person ids. 
            type = `dict[int, np.ndarray]`
        :return frames: a list of all frame indexs
        """
        dataset_dir_current = self.video_clip.dataset_dir
        order = self.video_clip.order

        csv_file_path = os.path.join(dataset_dir_current, 'true_pos_.csv')
        data = np.genfromtxt(csv_file_path, delimiter=',').T

        # Load data, and sort by person id
        persons = {}
        person_list = set(data.T[1].astype(np.int32))
        for person in person_list:
            index_current = np.where(data.T[1] == person)[0]
            persons[person] = np.column_stack([
                data[index_current, 0],
                data[index_current, 2 + order[0]],
                data[index_current, 2 + order[1]]])

        frames = list(set(data.T[0].astype(np.int32)))
        frames.sort()

        self.log('Load dataset {} done.'.format(csv_file_path))
        return persons, frames

    def process_metadata(self):
        """
        Process metadata of a video clip (like csv dataset 
        files) into numpy ndarray.
        """

        # make directories
        b = dir_check(os.path.join(dir_check('./dataset_npz'), self.name))
        npy_path = os.path.join(b, 'data.npz')

        # load from saved files
        if os.path.exists(npy_path):
            dat = np.load(npy_path, allow_pickle=True)
            neighbors = dat['neighbors']
            matrix = dat['matrix']
            frames = dat['frames']

        # or start processing and then saving
        else:
            persons_appear, frame_ids = self.load_csv(self.name)
            person_ids = list(persons_appear.keys())

            p = len(person_ids)
            f = len(frame_ids)

            # person_id -> person_index
            person_dict = dict(zip(person_ids, np.arange(p)))

            # frame_id -> frame_index
            frame_dict = dict(zip(frame_ids, np.arange(f)))

            # init the matrix
            matrix = INIT_POSITION * np.ones([f, p, 2])

            timebar = self.log_timebar(inputs=person_dict.items(),
                                       text='Processing dataset...',
                                       return_enumerate=False)
            for person_id, person_index in timebar:
                frame_id = persons_appear[person_id].T[0].astype(np.int32)
                frame_index = [frame_dict[fi] for fi in frame_id]

                matrix[frame_index, person_index, :] \
                    = persons_appear[person_id][:, 1:]

            neighbors = np.array([
                np.where(np.not_equal(data, INIT_POSITION))[0]
                for data in matrix[:, :, 0]], dtype=object)

            np.savez(npy_path,
                     neighbors=neighbors,
                     matrix=matrix,
                     frames=(frames := frame_ids))

        self.agent_count = matrix.shape[1]
        self.frame_number = matrix.shape[0]

        return neighbors, matrix, frames

    def make_trajectories(self):
        """
        Make trajectories from the processed dataset files.
        """
        if len(self.custom_list) == 3:
            self.neighbors, self.matrix, self.frames = self.custom_list
        else:
            self.neighbors, self.matrix, self.frames = self.process_metadata()

        trajs = []
        for person_index in range(self.agent_count):
            trajs.append(Trajectory(agent_index=person_index,
                                    trajectory=self.matrix[:, person_index, :],
                                    neighbors=self.neighbors,
                                    frames=self.frames,
                                    init_position=INIT_POSITION))

        self.trajectories = trajs
        return self

    def sample_train_data(self) -> list[Agent]:
        """
        Sampling train samples from trajectories.
        """

        if self.trajectories is None:
            self.make_trajectories()

        sample_rate, frame_rate = self.video_clip.paras
        frame_step = int(0.4 / (sample_rate / frame_rate))
        train_samples = []

        timebar = self.log_timebar(range(self.agent_count),
                                   text='Prepare train data...')
        for agent_index, _ in timebar:
            trajectory = self.trajectories[agent_index]
            start_frame = trajectory.start_frame
            end_frame = trajectory.end_frame

            for p in range(start_frame, end_frame, self.args.step * frame_step):
                # Normal mode
                if self.args.pred_frames > 0:
                    if p + (self.args.obs_frames + self.args.pred_frames) * frame_step > end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = p + (self.args.obs_frames +
                               self.args.pred_frames) * frame_step

                # Infinity mode, only works for destination models
                elif self.args.pred_frames == -1:
                    if p + (self.args.obs_frames + 1) * frame_step > end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = end_frame

                else:
                    raise ValueError(
                        '`pred_frames` should be a positive integer or -1.')

                train_samples.append(trajectory.sample(start_frame=p,
                                                       obs_frame=obs,
                                                       end_frame=end,
                                                       matrix=self.matrix,
                                                       frame_step=frame_step,
                                                       add_noise=False))

        return train_samples

    def make_maps(self, agents: list[Agent],
                  base_path: str,
                  save_map_file: str = None,
                  save_social_file: str = 'socialMap.npy',
                  save_para_file: str = 'para.txt',
                  save_centers_file: str = 'centers.txt'):
        """
        Make maps for input agents, and save them in the numpy format.

        :param agents: a list of agents that ready to calculate maps
        :param base_path: base folder to save the map and map parameters
        :param load_map_file: file name for the saved trajectory map (`.jpg` or `.png`).
        default is `None`. When this item is `None`, MapManager will build
        trajectory maps according to trajectories of the input agents.
        :param save_map_file: file name to save the built traj map
        :param save_social_file: file name to save the social map (already cut)
        :param save_para_file: file name to save the map parameters
        :param save_centers_file: path to save the centers
        """

        map_manager = MapManager(self.args, agents)

        if save_map_file:
            traj_map = map_manager.build_guidance_map(
                agents=agents,
                save=os.path.join(base_path, save_map_file))

        social_maps = []
        centers = []
        for agent in self.log_timebar(agents,
                                      'Build maps...',
                                      return_enumerate=False):

            centers.append(agent.traj[-1:, :])
            social_maps.append(map_manager.build_social_map(
                target_agent=agent,
                traj_neighbors=agent.get_pred_traj_neighbor_linear()))

        social_maps = np.array(social_maps)  # (batch, a, b)

        centers = np.concatenate(centers, axis=0)
        centers = map_manager.real2grid(centers)
        cuts = map_manager.cut_map(social_maps,
                                   centers,
                                   MAP_HALF_SIZE)
        paras = map_manager.real2grid_paras

        np.savetxt(os.path.join(base_path, save_centers_file), centers)
        np.savetxt(os.path.join(base_path, save_para_file), paras)
        np.save(os.path.join(base_path, save_social_file), cuts)


class DatasetManager(BaseObject):
    """
    DatasetsManager
    ---------------
    Manage all prediction training data.

    Public Methods
    --------------
    ```python
    # Prepare train agents from `DatasetManager`s
    (method) load_fromManagers: (self: DatasetsManager, dataset_managers: list[DatasetManager], mode='test') -> list[Agent]

    # Save and load agents' data
    (method) zip_and_save: (save_dir, agents: list[Agent]) -> None
    (method) load_and_unzip: (cls: Type[DatasetsManager], save_dir) -> list[Agent]
    ```
    """

    def __init__(self, args: Args):
        super().__init__()

        self.args = args
        self.dataset = Dataset(args.test_set)

    def load_from_videoClips(self, video_clips: list[VideoClipManager],
                             mode='test') -> list[Agent]:
        """
        Make or load train files to get train agents.
        (a list of agent managers, type = `Agent`)

        :param video_clips: a list of video clip managers (`VideoClipManager`)
        :return all_agents: a list of train agents (`Agent`)
        """
        all_agents = []
        count = 1
        total = len(video_clips)
        dir_check('./dataset_npz/')

        for clip in video_clips:
            print('({}/{})  Prepare test data in `{}`...'.format(
                count, total, clip.name))

            if (self.args.obs_frames, self.args.pred_frames) == (8, 12):
                data_path = './dataset_npz/{}/agent'.format(clip.name)
            else:
                data_path = './dataset_npz/{}/agent_{}to{}'.format(clip.name,
                                                                   self.args.obs_frames,
                                                                   self.args.pred_frames)

            endstring = '' if self.args.step == 4 else self.args.step
            data_path += '{}.npz'.format(endstring)

            if not os.path.exists(data_path):
                agents = clip.sample_train_data()
                self.zip_and_save(data_path, agents)
            else:
                agents = self.load_and_unzip(data_path)

            self.log('Successfully load train agents from `{}`'.format(data_path))

            if self.args.use_maps:
                map_path = dir_check(data_path.split('.np')[0] + '_maps')
                map_file = ('trajMap.png' if not self.args.use_extra_maps
                            else 'trajMap_load.png')

                try:
                    agents = self.load_maps(map_path, agents,
                                            map_file=map_file,
                                            social_file='socialMap.npy',
                                            para_file='para.txt',
                                            centers_file='centers.txt')

                except TrajMapNotFoundError:
                    path = os.path.join(map_path, map_file)
                    self.log(s := ('Trajectory map `{}`'.format(path) +
                                   ' not found, stop running...'),
                             level='error')
                    exit()

                except:
                    self.log('Load maps failed, start re-making...')

                    clip.make_maps(agents, map_path,
                                   save_map_file='trajMap.png',
                                   save_social_file='socialMap.npy',
                                   save_para_file='para.txt',
                                   save_centers_file='centers.txt')

                    agents = self.load_maps(map_path, agents,
                                            map_file=map_file,
                                            social_file='socialMap.npy',
                                            para_file='para.txt',
                                            centers_file='centers.txt')

                self.log('Successfully load maps from `{}`.'.format(map_path))

            all_agents += agents
            count += 1

        return all_agents

    def zip_and_save(self, save_dir, agents: list[Agent]):
        save_dict = {}
        for index, agent in enumerate(agents):
            save_dict[str(index)] = agent.zip_data()
        np.savez(save_dir, **save_dict)

    def load_and_unzip(self, save_dir) -> list[Agent]:
        save_dict = np.load(save_dir, allow_pickle=True)

        if save_dict['0'].tolist()['__version__'] < Agent.__version__:
            self.log(('Saved agent managers\' version is {}, ' +
                      'which is lower than current {}. Please delete ' +
                      'them and re-run this program, or there could ' +
                      'happen something wrong.').format(save_dict['0'].tolist()['__version__'],
                                                        Agent.__version__),
                     level='error')

        return [Agent().load_data(save_dict[key].tolist()) for key in save_dict.keys()]

    @classmethod
    def load(cls, args: Args, dataset: Union[str, list[str]], mode: str):
        """
        Load train samples in sub-dataset(s).

        :param args: args used
        :param dataset: dataset to load. Set it to `'auto'` to load train agents
        :param mode: load mode, canbe `'test'` or `'train'`
        :return agents: loaded agents. It returns a list of `[train_agents, test_agents]` when `mode` is `'train'`.
        """
        dir_check('./dataset_npz')
        Dm = cls(args)

        if dataset == 'auto':
            train_sets = Dm.dataset.train_sets
            test_sets = Dm.dataset.test_sets

            train_agents = cls.load(args, train_sets, mode='train')
            test_agents = cls.load(args, test_sets, mode='test')

            return train_agents, test_agents

        else:
            if type(dataset) == str:
                dataset = [dataset]

            dms = [VideoClipManager(args, d) for d in dataset]
            return Dm.load_from_videoClips(dms, mode=mode)

    def load_maps(self, base_path: str,
                  agents: list[Agent],
                  map_file: str,
                  social_file: str,
                  para_file: str,
                  centers_file: str) -> list[Agent]:
        """
        Load maps from the base folder

        :param base_path: base save folder
        :param agents: agents to assign maps
        :param map_file: file name for traj maps, support `.jpg` or `.png`
        :param social_file: file name for social maps, support `.npy`
        :param para_file: file name for map parameters, support `.txt`
        :param centers_file: file name for centers, support `.txt`

        :return agents: agents with maps
        """
        traj_map = cv2.imread(os.path.join(base_path, map_file))

        if traj_map is None:
            if self.args.use_extra_maps:
                raise TrajMapNotFoundError
            else:
                raise FileNotFoundError

        traj_map = (traj_map[:, :, 0]).astype(np.float32)/255.0

        social_map = np.load(os.path.join(
            base_path, social_file), allow_pickle=True)
        para = np.loadtxt(os.path.join(base_path, para_file))
        centers = np.loadtxt(os.path.join(base_path, centers_file))

        batch_size = len(social_map)
        traj_map = np.repeat(traj_map[np.newaxis, :, :], batch_size, axis=0)
        traj_map_cut = MapManager.cut_map(traj_map,
                                          centers,
                                          MAP_HALF_SIZE)

        for agent, t_map, s_map in zip(agents, traj_map_cut, social_map):
            Agent.set_map(agent, 0.5*t_map + 0.5*s_map, para)

        return agents
