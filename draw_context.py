# draw_context.py
import random

class DrawContext:
    def __init__(self, *, nres, probzero, current_nca, current_nco,
                 nposs, poss, nmatch, possmatch,
                 used_nca, deg_nca, used_nco, deg_nco,
                 fix, overuse, skip_mc):
        self.nres        = nres
        self.probzero    = probzero
        # Pad current_nca/current_nco to length nres+1
        need = nres + 1
        if len(current_nca) < need:
            current_nca.extend([0] * (need - len(current_nca)))
        if len(current_nco) < need:
            current_nco.extend([0] * (need - len(current_nco)))
        self.current_nca = current_nca
        self.current_nco = current_nco

        self.nposs       = nposs
        self.poss        = poss
        self.nmatch      = nmatch
        self.possmatch   = possmatch
        self.used_nca    = used_nca
        self.deg_nca     = deg_nca
        self.used_nco    = used_nco
        self.deg_nco     = deg_nco
        self.fix         = fix
        self.overuse     = overuse
        self.skip_mc     = skip_mc

        # Outputs
        self.kres         = 0
        self.asgn_oldnca  = 0
        self.asgn_oldnco  = 0
        self.asgn_newnca  = 0
        self.asgn_newnco  = 0
        self.nused_oldnca = 0
        self.nused_oldnco = 0
        self.nused_newnca = 0
        self.nused_newnco = 0

    def draw(self):
        while True:
            # 1) pick kres in [1, nres]
            while True:
                kres = random.randint(1, self.nres)
                # now kres is safe to index into current_nca/current_nco
                if self.skip_mc[kres] == 0 and self.nposs[kres] > 0:
                    break

            old_nca = self.current_nca[kres]
            old_nco = self.current_nco[kres]
            self.nused_oldnca = int(old_nca != 0)
            self.nused_oldnco = int(old_nco != 0)

            # 2) propose new assignments
            if random.random() < self.probzero:
                new_nca = 0
                new_nco = 0
                self.nused_newnca = 0
                self.nused_newnco = 0
            else:
                choice   = random.randint(1, self.nposs[kres])
                new_nca  = self.poss[kres][choice]
                self.nused_newnca = 1

                if self.deg_nca[new_nca] <= self.used_nca[new_nca]:
                    new_nca = 0
                    new_nco = 0
                    self.nused_newnca = self.nused_newnco = 0
                elif self.nmatch[new_nca] == 0:
                    new_nco = 0
                    self.nused_newnco = 0
                else:
                    m = random.randint(1, self.nmatch[new_nca])
                    if self.fix[new_nca][m] < 0:
                        over = any(
                            self.deg_nco[s] <= self.used_nco[s]
                            for s in self.overuse[new_nca][m]
                        )
                        if over:
                            new_nco = 0
                            self.nused_newnco = 0
                        else:
                            new_nco = m
                            self.nused_newnco = 1
                    else:
                        new_nco = m
                        self.nused_newnco = 1

            # 3) accept only if changed
            if new_nca != old_nca or new_nco != old_nco:
                break

        # store
        self.kres         = kres
        self.asgn_oldnca  = old_nca
        self.asgn_oldnco  = old_nco
        self.asgn_newnca  = new_nca
        self.asgn_newnco  = new_nco

