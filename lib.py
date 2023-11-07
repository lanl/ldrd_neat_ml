# these features were considered potentially interesting
# for machine learning-based prediction of polymer
# properties by Mihee, Cesar, and Tyler during a discussion
# on Nov. 3/2023
features_of_interest = ["volume_fraction",
                        # temperature at which light transmission of
                        # polymer mixture changes (within biological range):
                        "temp_light_transmission", 
                        # polydispersity index (dispersity) -- different
                        # polymer lengths that result from imperfections in
                        # human synthesis/manufacturing processes:
                        "PDI",
                        # weight-average molar mass:
                        "M_w",
                        "backbone_or_torsion_angles",
                        "radius_of_gyration",
                        # these maps might be intra or even inter-molecular:
                        "contact_or_adjacency_maps",
                        # H-bond contacts in first solvation shell?
                        "h_bond_contacts",
                        # Mihee can measure interfacial tension
                        "interfacial_tension",
                        # Viscoscity may matter for cytometry, but not clear
                        # if it would matter for polymer phase separatation/
                        # microparticle formation:
                        "viscosity"]
