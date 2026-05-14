import ot
import torch


def gwgrad_partial(C1, C2, T):
    """Compute the GW gradient. Note: we can not use the trick in :ref:`[12] <references-gwgrad-partial>`
    as the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        gradient


    .. _references-gwgrad-partial:
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    device = C1.device
    cC1 = torch.matmul(C1 ** 2 / 2, torch.matmul(T, torch.ones(C2.shape[0], device=device).reshape(-1, 1)))
    cC2 = torch.matmul(torch.matmul(torch.ones(C1.shape[0], device=device).reshape(1, -1), T), C2 ** 2 / 2)
    constC = cC1 + cC2
    A = -torch.matmul(C1, T).matmul(C2.T)
    tens = constC + A
    return tens * 2


def gwgrad_partial_with_prior(C1, C2, T, prior):
    """Compute the GW gradient. Note: we can not use the trick in :ref:`[12] <references-gwgrad-partial>`
    as the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        gradient


    .. _references-gwgrad-partial:
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    device = C1.device
    cC1 = torch.matmul(C1 ** 2 / 2, torch.matmul(T, torch.ones(C2.shape[0], device=device).reshape(-1, 1)))
    cC2 = torch.matmul(torch.matmul(torch.ones(C1.shape[0], device=device).reshape(1, -1), T), C2 ** 2 / 2)
    constC = cC1 + cC2
    A = -torch.matmul(C1, T).matmul(C2.T)
    tens = constC + A
    return tens * 2 * prior



def gwloss_partial(C1, C2, T):
    """Compute the GW loss.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    GW loss
    """
    g = gwgrad_partial(C1, C2, T) * 0.5
    return torch.sum(g * T)


def gwloss_partial_with_prior(C1, C2, T, prior):
    """Compute the GW loss.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    GW loss
    """
    g = gwgrad_partial_with_prior(C1, C2, T, prior) * 0.5
    return torch.sum(g * T)


def partial_gromov_wasserstein_with_prior(C1, C2, p, q, prior=None, m=None, nb_dummies=1, G0=None, numItermax=1000, tol=1e-7,
                               log=False, verbose=False, **kwargs):
    device = p.device

    if prior is None:
        prior = torch.ones((len(p), len(q)), device=device)
    
    with torch.no_grad():
        if m is None:
            m = torch.min(torch.sum(p), torch.sum(q))
        elif m < 0:
            raise ValueError("Problem infeasible. Parameter m should be greater"
                            " than 0.")
        elif m > torch.min(torch.sum(p), torch.sum(q)):
            raise ValueError("Problem infeasible. Parameter m should lower or"
                            " equal than min(|a|_1, |b|_1).")

        if G0 is None:
            G0 = torch.outer(p, q)

        dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
        q_extended = torch.cat((q, torch.tensor([(torch.sum(p) - m) / nb_dummies] * nb_dummies, device=device)), dim=0)
        p_extended = torch.cat((p,  torch.tensor([(torch.sum(q) - m) / nb_dummies] * nb_dummies, device=device)),  dim=0)

        cpt = 0
        err = 1

        if log:
            log = {'err': []}

        while (err > tol and cpt < numItermax):

            Gprev = G0.clone()

            M = gwgrad_partial_with_prior(C1, C2, G0, prior)
            M_emd = torch.zeros(dim_G_extended, device=device)
            M_emd[:len(p), :len(q)] = M
            M_emd[-nb_dummies:, -nb_dummies:] = torch.max(M) * 1e2

            Gc, logemd = ot.emd(p_extended, q_extended, M_emd, log=True, **kwargs)

            if logemd['warning'] is not None:
                raise ValueError("Error in the EMD resolution: try to increase the"
                                " number of dummy points")

            G0 = Gc[:len(p), :len(q)]

            if cpt % 10 == 0:  # to speed up the computations
                err = torch.norm(G0 - Gprev)
                if log:
                    log['err'].append(err)
                if verbose:
                    if cpt % 200 == 0:
                        print('{:5s}|{:12s}|{:12s}'.format(
                            'It.', 'Err', 'Loss') + '\n' + '-' * 31)
                    print('{:5d}|{:8e}|{:8e}'.format(cpt, err,
                                                    gwloss_partial_with_prior(C1, C2, G0, prior)))

            deltaG = G0 - Gprev
            a = gwloss_partial_with_prior(C1, C2, deltaG, prior) # (M(Cs, Ct) O E) * prior
            b = 2 * torch.sum(M * deltaG) # prior에 두번 곱해져있었음;;
            if b > 0:  # due to numerical precision
                gamma = 0
                cpt = numItermax
            elif a > 0:
                gamma = min(1, torch.divide(-b, 2.0 * a))
            else:
                if (a + b) < 0:
                    gamma = 1
                else:
                    gamma = 0
                    cpt = numItermax

            G0 = Gprev + gamma * deltaG
            cpt += 1

        if log:
            log['partial_gw_dist'] = gwloss_partial_with_prior(C1, C2, G0, prior)
            return G0[:len(p), :len(q)], log
        else:
            return G0[:len(p), :len(q)]

