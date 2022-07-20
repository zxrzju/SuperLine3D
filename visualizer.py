import numpy as np
import os
import ntpath
import time



class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id

        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(env='%d' % self.display_id)



    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals):
        if self.display_id > 0: # show images in the browser
            idx = 1
            for label, item in visuals.items():
                if 'data_vis' in label:
                    data_vis_np, data_vis_color_np = item
                    self.vis.scatter(data_vis_np,
                                     Y=None,
                                     opts=dict(title=label,
                                               markersize=4,
                                               markercolor=data_vis_color_np,
                                               markersymbol='circle'),
                                     win=self.display_id + idx,
                                     name='data_vis')
                elif 'pc' in label:
                    N = item.shape[0]
                    color_np = np.repeat(np.expand_dims(np.array([125, 125, 125], dtype=np.int64), axis=0),
                                         N,
                                         axis=0)  # 1x3 -> Nx3
                    self.vis.scatter(item,
                                     Y=None,
                                     opts=dict(title=label,
                                               markersize=4,
                                               markercolor=color_np,
                                               markersymbol='circle'),
                                     win=self.display_id + idx)
                elif 'img' in label:
                    # the transpose: HxWxC -> CxHxW
                    self.vis.image(np.transpose(item, (2,0,1)), opts=dict(title=label),
                                   win=self.display_id + idx)
                idx += 1

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        # clamp the errors at plot, to increase resolution
        # for key, value in errors.items():
        #     if value > 1:
        #         errors[key] = 1

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)


