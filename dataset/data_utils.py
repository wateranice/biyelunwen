import torch
from torchvision import datasets, transforms
import numpy as np


def _get_transforms(dataset: str):
    name = (dataset or "cifar10").lower()
    if name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if name == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    if name in ("fmnist", "fashion_mnist", "fashion-mnist"):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
    raise ValueError(f"未知 dataset={dataset!r}，支持: cifar10, mnist, fmnist")


def _load_train_test(dataset: str, data_root: str):
    name = (dataset or "cifar10").lower()
    tfm = _get_transforms(name)
    if name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm)
        n_classes = 10
    elif name == "mnist":
        train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
        n_classes = 10
    elif name in ("fmnist", "fashion_mnist", "fashion-mnist"):
        train_ds = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=tfm)
        n_classes = 10
    else:
        raise ValueError(f"未知 dataset={dataset!r}")
    return train_ds, test_ds, n_classes


def _split_dirichlet(label_indices, n_clients, beta, n_classes):
    client_id_map = [[] for _ in range(n_clients)]
    for k in range(n_classes):
        idx_k = label_indices[k].copy()
        if len(idx_k) == 0:
            continue
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet([beta] * n_clients)
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        split_idx = np.split(idx_k, proportions)
        for i in range(n_clients):
            client_id_map[i].extend(np.asarray(split_idx[i]).ravel().tolist())
    return client_id_map


def _split_pathological(label_indices, n_clients, n_classes, seed):
    """
    病态 / 标签倾斜划分：将类别集合打乱后尽量均分给各客户端，
    每个客户端只包含分配给它的若干类样本（类间不重叠）。
    若 num_clients > n_classes，会出现空客户端或极少样本，建议在 YAML 中保持 num_clients <= n_classes。
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    perm = rng.permutation(n_classes)
    splits = np.array_split(perm, n_clients)
    client_id_map = [[] for _ in range(n_clients)]
    for cid, labs in enumerate(splits):
        for lab in labs.tolist():
            client_id_map[cid].extend(label_indices[int(lab)].tolist())
    return client_id_map


def prepare_data(
    n_clients=10,
    beta=0.5,
    proxy_ratio=0.04,
    data_root="./data",
    seed=None,
    dataset="cifar10",
    partition="dirichlet",
    proxy_disjoint_from_clients: bool = False,
):
    """
    加载 ``dataset``（cifar10 / mnist / fmnist），按 ``partition``（dirichlet / pathological）划分客户端索引，
    并划出 ``proxy_ratio`` 比例的代理子集。

    ``partition=pathological`` 时 ``beta`` 不参与划分；仍建议在 YAML 里保留 ``dirichlet_beta`` 以便切回 dirichlet 时少改字段。

    ``proxy_disjoint_from_clients=True`` 时：先用与 ``random_split`` 同源的 ``torch.Generator(seed)``
    无放回抽取代理索引，再在**剩余**训练索引上构造 ``label_indices`` 并划分客户端，保证代理样本
    不会进入任一客户端本地训练子集（集合互斥）。为 ``False`` 时保持旧行为（代理与客户端独立随机，
    索引可能重叠）。
    """
    name = (dataset or "cifar10").lower()
    part = (partition or "dirichlet").lower()

    if not 0.0 < proxy_ratio < 1.0:
        raise ValueError("proxy_ratio 必须在 (0, 1) 内")

    train_ds, test_ds, n_classes = _load_train_test(name, data_root)

    n_train = len(train_ds)
    proxy_size = int(n_train * proxy_ratio)
    proxy_size = max(1, min(proxy_size, n_train - 1))

    y_train = np.array(train_ds.targets)

    if proxy_disjoint_from_clients:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        perm = torch.randperm(n_train, generator=g).numpy()
        proxy_idx_set = set(perm[:proxy_size].tolist())
        not_proxy = np.ones(n_train, dtype=bool)
        for _pi in proxy_idx_set:
            not_proxy[_pi] = False
        label_indices = [np.where((y_train == c) & not_proxy)[0] for c in range(n_classes)]
        if part == "dirichlet":
            client_id_map = _split_dirichlet(label_indices, n_clients, beta, n_classes)
        elif part in ("pathological", "pathology", "shard"):
            client_id_map = _split_pathological(label_indices, n_clients, n_classes, seed)
        else:
            raise ValueError(f"未知 partition={partition!r}，支持: dirichlet, pathological")
        proxy_ds = torch.utils.data.Subset(train_ds, sorted(proxy_idx_set))
    else:
        label_indices = [np.where(y_train == i)[0] for i in range(n_classes)]
        if part == "dirichlet":
            client_id_map = _split_dirichlet(label_indices, n_clients, beta, n_classes)
        elif part in ("pathological", "pathology", "shard"):
            client_id_map = _split_pathological(label_indices, n_clients, n_classes, seed)
        else:
            raise ValueError(f"未知 partition={partition!r}，支持: dirichlet, pathological")

        generator = None
        if seed is not None:
            g2 = torch.Generator()
            g2.manual_seed(int(seed))
            generator = g2

        proxy_ds, _ = torch.utils.data.random_split(
            train_ds, [proxy_size, n_train - proxy_size], generator=generator
        )

    for i, idxs in enumerate(client_id_map):
        if len(idxs) == 0:
            raise ValueError(
                f"客户端 {i} 在 {part} 划分下无训练样本；请减小 num_clients 或调整 "
                f"dirichlet_beta / pathological 设置（pathological 建议 num_clients <= {n_classes}）。"
            )

    return train_ds, test_ds, proxy_ds, client_id_map
