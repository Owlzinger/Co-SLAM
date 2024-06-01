import torch

if __name__ == '__main__':
    config = 5
    ids = torch.tensor([0, 5, 10])
    frame_ids = torch.tensor([0, 5, 10, 15, 20])
    idx_cur = [30, 35, 40]
    # ids 2048
    # idx,cur 1024
    #   2
    ids_all2 = torch.cat([ids // config, -torch.ones((len(idx_cur)))]).to(torch.int64)
    # ids_all: tensor([ 0,  1,  2,  3,  4, -1, -1, -1], dtype=torch.int64)

    #   3
    ids_all = torch.cat([
        torch.tensor(
            [frame_ids.tolist().index(id) if id in frame_ids else -1 for id in ids]),
        -torch.ones((len(idx_cur)))
    ]).to(torch.int64)
    # ids_all: tensor([ 0,  1,  2,  3,  4, -1, -1, -1], dtype=torch.int64)
    print("done")
