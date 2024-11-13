# Copyright © 2024 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
"""
This example simulates a two-node secure key distribution setup.
An entangled photon source transmits entangled photons to two receiving nodes, Alice, and Bob.
The BBM92 QKD (quantum key distribution) protocol is used to establish a shared key between two of the three nodes.
For more details on the BBM92 protocol including a description of the protocol steps
and the implementation in this example, please see `example_qkd_bbm92.py`.

This example assumes an entangled photon source emitting entangled photon pairs with wavelength
equal to 810 nm. Realistic visibilites and source rates for Aliro's equipment
are assumed. The loss per link, in dB, can be adjusted according to expected experimental losses,
by modifying `LINK_LOSS_IN_DB`. The minimum
time resolution, corresponding to the time resolution of the time tagger or FPGA, is set to
a current common best resolution.

When the example is run, two entangled photons are routed from the entangled photon source Phoebe to the
two nodes Alice and Bob. The two nodes are also connected through classical
channels, so they can communicate their basis results. When the example finishes running,
the two sifted keys will be logged and the QBER and sifted key rates will also be logged.
"""

from enum import Enum, auto
from functools import partial
from typing import List

import random
import numpy as np
import simpy

import aqnsim
from aqnsim import SECOND, TERAHERTZ, SPEED_OF_LIGHT, NANOSECOND, DEFAULT_REFRACTIVE_INDEX


BASIS_CHOICE_KEY = "basis_choice"  # key indexing choice of basis in classical messages
MEASUREMENTS_COMPLETE_SIGNAL_NAME = "measurements_complete"
CLASSICAL_MSG_RECEIVED_SIGNAL_NAME = "classical_message_received"

# Simulation parameters
CHANNEL_LENGTH = 1  # 1 in meters
CHANNEL_DELAY = CHANNEL_LENGTH * DEFAULT_REFRACTIVE_INDEX / SPEED_OF_LIGHT  # latency for quantum and classical links
COLLECTION_TIME = 1e-3 * SECOND  # Time to collect counts at each detector
SIMULATION_TIME = COLLECTION_TIME*1.1
LINK_LOSS_IN_DB = 0 # Loss per link, in dB (NOTE: This will be close to 7 dB for the real experiment)

# Source parameters
SINGLES_RATE_A = 306400 / SECOND # Singles count rate from port A of source (Provided by OZ Optics)
SINGLES_RATE_B = 237400 / SECOND # Singles count rate from port B of source (Provided by OZ Optics)
SOURCE_PAIR_RATE = 1e5 / SECOND  # Approx counts per second from source (Provided by OZ Optics)
SOURCE_VISIBILITY = 0.98  # Visibility of the entangled source
SOURCE_EST_QBER = (1 - SOURCE_VISIBILITY) / 2  # Estimated QBER
SOURCE_WAVELENGTH = 810 * 10**-9 # Wavelength of source photons, in meters
SOURCE_BANDWIDTH_FWHM = 3 * 10**-9 # Source bandwidth FWHM, in meters
SOURCE_BANDWITH_ST_DEV = SOURCE_BANDWIDTH_FWHM / (2 * np.sqrt(2 * np.log(2)))
SOURCE_FREQ = SPEED_OF_LIGHT / SOURCE_WAVELENGTH
SOURCE_FREQ_WIDTH = SPEED_OF_LIGHT * SOURCE_BANDWITH_ST_DEV / (SOURCE_WAVELENGTH**2)

# Detector parameters (SPADs)
DETECTOR_DARK_COUNTS = {0: 0, 1: 250 / SECOND}  # Average dark counts per detector
DETECTOR_JITTER = 1e-9 * SECOND  # Jitter of detector
DETECTOR_DEAD_TIME = 45e-9 * SECOND  # Dead time of detector
DETECTOR_FREQ_WIDTH = SPEED_OF_LIGHT / 500e-9 - SPEED_OF_LIGHT / 700e-9
DETECTOR_EFFICIENCY = {
    aqnsim.MAX_EFFICIENCY_KEY: 0.60,
    aqnsim.FREQUENCY_PARAM_KEY: SOURCE_FREQ,
    aqnsim.FREQUENCY_WIDTH_PARAM_KEY: DETECTOR_FREQ_WIDTH
}  # Efficiency of detector
MINIMUM_TIME_RESOLUTION = 1 * NANOSECOND  # Time resolution of the FPGA


# possible basis choices used by the receiving nodes
class BasisChoices(Enum):
    HV_BASIS = 0  # Z-basis, Horizontal/Vertical
    DA_BASIS = auto()  # X-basis, Diagonal/Anti-diagonal


"""
SendProtocol: used by Phoebe to send entangled photon pairs with one photon sent to
each receiving node
"""


class SendProtocol(aqnsim.NodeProtocol):
    """
    Node protocol for sending entangled photon pairs
    :param env: The simulation environment to use
    :param qs: The QuantumSimulator to use for simulating quantum behavior
    :param node: The node on which the protocol acts
    :param name: The name of the protocol
    """

    def __init__(self, env: simpy.Environment, qs: aqnsim.QuantumSimulator, node: aqnsim.Node, name: str = None):
        super().__init__(env=env, node=node, name=name)

        self.qs = qs

    def _send_measurements_complete_message(self):
        """
        Send a message to the receiving nodes indicating that all measurements have been completed
        """
        msg = aqnsim.CMessage(
            sender=self.name,
            action="MEASUREMENTS_COMPLETE",
            status=aqnsim.StatusMessages.SUCCESS,
        )
        self.node.ports["classical_channel_port"].rx_output(msg)

    @aqnsim.process
    def run(self):
        """Main run method of the SendProtocol--sends out multiple entangled photon
        pairs to the receiving nodes.
        """
        # send out entangled photons to be used in generating QKD key
        self.node.subcomponents["entangled_photon_source"].turn_on()

        # Wait for COLLECTION_TIME to collect coincidence counts
        yield self.wait(COLLECTION_TIME)

        # Turn off source
        self.node.subcomponents["entangled_photon_source"].turn_off()

        # Send message to nodes to indicate measurement is complete
        self._send_measurements_complete_message()


"""
ReceiveProtocol: used by the nodes receiving entangled photons and establishing a shared
secret key.
"""


class ReceiveProtocol(aqnsim.NodeProtocol):
    """
    Node protocol used by Alice, and Bob. The nodes receive one photon
    from each entangled pair, measure the photons,
    and use the measurements to generate a shared secret key.
    :param env: The simulation environment to use
    :param qs: The QuantumSimulator to use for simulating quantum behavior
    :param node: The node on which the protocol acts
    :param name: The name of the protocol
    """

    def __init__(self, env: simpy.Environment, qs: aqnsim.QuantumSimulator, node: aqnsim.Node, name: str = None):
        super().__init__(env=env, node=node, name=name)  # sets self.env, self.node

        # track the quantum simulator instance
        self.qs = qs

        # set up classical channel input handler
        self.node.ports["classical_channel_port"].add_rx_input_handler(handler=self.input_handler_classical)

        # set up some measurement handlers to process results from the photon detectors
        measurement_handler_HV_0 = partial(self._measurement_handler, basis=BasisChoices.HV_BASIS, detection=0)
        measurement_handler_HV_1 = partial(self._measurement_handler, basis=BasisChoices.HV_BASIS, detection=1)
        measurement_handler_DA_0 = partial(self._measurement_handler, basis=BasisChoices.DA_BASIS, detection=0)
        measurement_handler_DA_1 = partial(self._measurement_handler, basis=BasisChoices.DA_BASIS, detection=1)
        self.node.subcomponents[f"{node.name}_detector_0_0"].ports["cout0"].add_rx_output_handler(
            measurement_handler_HV_0
        )
        self.node.subcomponents[f"{node.name}_detector_0_1"].ports["cout0"].add_rx_output_handler(
            measurement_handler_HV_1
        )
        self.node.subcomponents[f"{node.name}_detector_1_0"].ports["cout0"].add_rx_output_handler(
            measurement_handler_DA_0
        )
        self.node.subcomponents[f"{node.name}_detector_1_1"].ports["cout0"].add_rx_output_handler(
            measurement_handler_DA_1
        )

        # create signals
        self.add_signal(MEASUREMENTS_COMPLETE_SIGNAL_NAME)
        self.add_signal(CLASSICAL_MSG_RECEIVED_SIGNAL_NAME)

        # initialize tracking variables
        self.measurements = {}
        self.basis_choices = {}
        self.sifted_key = []
        self.classical_msg = None

        # Create a GaussianDelayModel to model detector jitter
        self.jitter_model = aqnsim.GaussianDelayModel(mean=0, std=DETECTOR_JITTER)

    def input_handler_classical(self, msg: aqnsim.CMessage):
        """Handler for messages communicated over the classical channel"""
        if not isinstance(msg, aqnsim.CMessage):
            aqnsim.simlogger.warning(
                f"classical channel input handler at {self.node.name} received message with unexpected data type"
            )
        classical_msg_action = msg.action
        if classical_msg_action == "MEASUREMENTS_COMPLETE":
            # Stop processing new photodetector counts
            node_name = self.node.name
            self.node.subcomponents[f"{node_name}_detector_0_0"].ports["cout0"].rx_output_handlers = []
            self.node.subcomponents[f"{node_name}_detector_0_1"].ports["cout0"].rx_output_handlers = []
            self.node.subcomponents[f"{node_name}_detector_1_0"].ports["cout0"].rx_output_handlers = []
            self.node.subcomponents[f"{node_name}_detector_1_1"].ports["cout0"].rx_output_handlers = []

            # send a signal that all measurements have been completed
            self.send_signal(MEASUREMENTS_COMPLETE_SIGNAL_NAME)
        elif classical_msg_action == "TRANSMIT_BASIS":
            self.classical_msg = msg.content["basis_choices"]
            self.send_signal(CLASSICAL_MSG_RECEIVED_SIGNAL_NAME)
        else:
            raise ValueError(f"Unexpected message action {classical_msg_action} received.")

    def _measurement_handler(self, msg, basis: BasisChoices, detection):
        """
        Measurement handler to process measurement results from the photon detectors
        :param msg: classical message containing measurement result
        :param basis: identifies which basis the detection was performed in
        :param detection: the detected polarization
        """
        aqnsim.simlogger.info(f"measurement_handler at node {self.node.name}, msg: {msg}")

        # Add detector jitter; draw from a Gaussian distribution with sigma = RMS and mean = 0
        jitter = self.jitter_model.get_delay()

        # Create timestamp
        timestamp = MINIMUM_TIME_RESOLUTION * round((self.env.now + jitter) / MINIMUM_TIME_RESOLUTION)

        self.measurements[timestamp] = detection
        self.basis_choices[timestamp] = basis

    def _process_classical_message(self):
        """Method for processing classical messages from the peer node"""
        # process classical message related to basis choices to obtain a raw sifted key
        # note: the receiving nodes both measure before sending their basis choices, so
        #       self.basis_choices should always be ready before this method is called
        if BASIS_CHOICE_KEY in self.classical_msg:
            common_basis_choice_times = []
            remote_basis_choices = self.classical_msg[BASIS_CHOICE_KEY]

            for timestamp, remote_basis_choice in remote_basis_choices.items():
                # If the timetagged basis choice at Alice matches the time tagged basis choice at Bob,
                # Append the common basis choice to a list.
                if self.basis_choices.get(timestamp) == remote_basis_choice:
                    common_basis_choice_times.append(timestamp)

            M = len(common_basis_choice_times)
            self.sifted_key = [self.measurements[common_basis_choice_times[k]] for k in range(M)]
            aqnsim.simlogger.info(f"SIFTED KEY obtained at node {self.node.name}: " f"{self.sifted_key}")
            assert len(self.sifted_key) > 0

    def _send_basis_choices(self):
        """
        Send our list of basis choices to our peer node
        """
        basis_choices = {BASIS_CHOICE_KEY: self.basis_choices}
        msg = aqnsim.CMessage(
            sender=self.name,
            action="TRANSMIT_BASIS",
            status=aqnsim.StatusMessages.SUCCESS,
            content={"basis_choices": basis_choices},
        )
        self.node.ports["classical_channel_port"].rx_output(msg)

    @aqnsim.process
    def run(self):
        """Main run method of the protocol"""
        yield self.await_signal(self, MEASUREMENTS_COMPLETE_SIGNAL_NAME)
        self._send_basis_choices()
        yield self.await_signal(self, CLASSICAL_MSG_RECEIVED_SIGNAL_NAME)
        self._process_classical_message()


"""
Set up the network nodes, links, optical components, and protocols. The internal structure
of the nodes is as follows:

Send Node (Phoebe)
    EntangledPolarizationSource

Receiving Node (Alice, Bob)
    Beamsplitter (BS)
    BS Outport 0:
        PolarizingBeamSplitter (PBS)
            Detector 0
            Detector 1
    BS Outport 1
        Half-Waveplate (HWP)
        PolarizingBeamSplitter (PBS)
            Detector 0
            Detector 1

"""


def setup_network(env: simpy.Environment, qs: aqnsim.QuantumSimulator):
    """Sets up the network for the BBM92 protocol
    :param env: The simulation environment used for the example
    :param qs: The quantum simulator for simulating quantum physics
    """
    # Instantiate nodes and network that contains them
    alice = aqnsim.Node(env=env, name="Alice")  # receives one side of entangled photon pair
    bob = aqnsim.Node(env=env, name="Bob")  # receives other side of entangled photon pair
    phoebe = aqnsim.Node(env=env, name="Phoebe")  # produces entangled photon pairs
    network = aqnsim.Network(env=env, qs=qs, nodes=[alice, bob, phoebe])

    _setup_source_node(qs, phoebe)  # set up the entanglement source

    # set up receiving nodes--Alice and Bob each receive one side of an entangled photon pair
    receiving_nodes = [alice, bob]
    for node in receiving_nodes:
        _setup_receiving_node(qs, node)

    # create fiber links for distributing entangled photons
    fiber_link_alice = aqnsim.FiberLink(env=env, qs=qs, length=CHANNEL_LENGTH, insertion_losses=LINK_LOSS_IN_DB)
    fiber_link_bob = aqnsim.FiberLink(env=env, qs=qs, length=CHANNEL_LENGTH, insertion_losses=LINK_LOSS_IN_DB)

    # create classical links for coordination with source
    classical_link_phoebe_alice = aqnsim.ClassicalLink(env=env, delay=CHANNEL_DELAY, name="classical_link_phoebe_alice")
    classical_link_phoebe_bob = aqnsim.ClassicalLink(env=env, delay=CHANNEL_DELAY, name="classical_link_phoebe_bob")
    classical_link_bob_alice = aqnsim.ClassicalLink(env=env, delay=CHANNEL_DELAY, name="classical_link_bob_alice")

    # connect network links
    network.add_link(
        link=fiber_link_alice,
        node1=phoebe,
        node2=alice,
        node1_portname="entangled_photon_output_0_port",
        node2_portname="entangled_photon_input_port",
    )
    network.add_link(
        link=fiber_link_bob,
        node1=phoebe,
        node2=bob,
        node1_portname="entangled_photon_output_1_port",
        node2_portname="entangled_photon_input_port",
    )
    network.add_link(
        link=classical_link_bob_alice,
        node1=alice,
        node2=bob,
        node1_portname="classical_channel_port",
        node2_portname="classical_channel_port",
    )
    network.add_link(
        link=classical_link_phoebe_alice,
        node1=alice,
        node2=phoebe,
        node1_portname="classical_channel_port",
        node2_portname="classical_channel_port",
    )
    network.add_link(
        link=classical_link_phoebe_bob,
        node1=bob,
        node2=phoebe,
        node1_portname="classical_channel_port",
        node2_portname="classical_channel_port",
    )

    # attach protocols to the nodes
    SendProtocol(env=env, qs=qs, node=phoebe)
    ReceiveProtocol(env=env, qs=qs, node=alice)
    ReceiveProtocol(env=env, qs=qs, node=bob)

    return network


def _setup_source_node(qs, node):
    """
    Set up a node that emits entangled Werner states encoded in polarization photons.
    Entangled pairs are emitted based on a clock with an exponential delay model.

    :param qs: A quantum simulator instance.
    :param node: The node that will have the entanglement source added to it.
    """
    # set up entangled photon source node (Phoebe)
    node.add_port(port_name="entangled_photon_output_0_port")
    node.add_port(port_name="entangled_photon_output_1_port")
    node.add_port(port_name="classical_channel_port")

    # Set up the source clock, which should follow an exponential delay model
    source_clock = aqnsim.Clock(node.env, aqnsim.ExponentialDelayModel(lam=SOURCE_PAIR_RATE), tick_at_start=False)

    # Different mechanisms can lower visibility; using `get_noisy_werner` mixes the Bell state with the maximally
    # mixed state to be conservative.
    state_distribution = [(1, aqnsim.get_noisy_werner(state="phi_plus", fidelity=1 - SOURCE_EST_QBER / 2))]
    state_model = aqnsim.StateModel(
        state_distribution=state_distribution, formalism=aqnsim.StateFormalisms.DENSITY_MATRIX
    )
    entangled_photon_source = aqnsim.EntangledPolarizationSource(
        env=node.env,
        qs=qs,
        state_model=state_model,
        name="entangled_photon_source",
        mode_shape=aqnsim.GaussianModeShape(frequency=SOURCE_FREQ, frequency_width=SOURCE_FREQ_WIDTH),
        clock=source_clock,
    )
    entangled_photon_source.ports["qout0"].forward_output_to_output(node.ports["entangled_photon_output_0_port"])
    entangled_photon_source.ports["qout1"].forward_output_to_output(node.ports["entangled_photon_output_1_port"])
    node.add_subcomponent(entangled_photon_source)

    noise_source_clock_A = aqnsim.Clock(node.env, aqnsim.ExponentialDelayModel(lam=SINGLES_RATE_A), tick_at_start=False)
    noise_source_A = aqnsim.SinglePhotonSource(env=node.env,
                                             qs=qs,
                                             clock=noise_source_clock_A,
                                             multi_photon=False,
                                             name='noise_source_A',
                                             mode_shape=aqnsim.GaussianModeShape(frequency=SOURCE_FREQ, frequency_width=SOURCE_FREQ_WIDTH),
                                             )

    noise_source_clock_B = aqnsim.Clock(node.env, aqnsim.ExponentialDelayModel(lam=SINGLES_RATE_B), tick_at_start=False)
    noise_source_B = aqnsim.SinglePhotonSource(env=node.env,
                                             qs=qs,
                                             clock=noise_source_clock_B,
                                             multi_photon=False,
                                             name='noise_source_B',
                                             mode_shape=aqnsim.GaussianModeShape(frequency=SOURCE_FREQ, frequency_width=SOURCE_FREQ_WIDTH),
                                             )
    noise_source_A.ports["qout0"].forward_output_to_output(node.ports["entangled_photon_output_0_port"])
    noise_source_B.ports["qout0"].forward_output_to_output(node.ports["entangled_photon_output_1_port"])
    node.add_subcomponent(noise_source_A)
    node.add_subcomponent(noise_source_B)


def _setup_receiving_node(qs, node):
    # add port for receiving photons and another for classical communication
    node.add_port(port_name="entangled_photon_input_port")
    node.add_port(port_name="classical_channel_port")

    # create a beamsplitter to process input photons, the output port of the processed photon
    # effectively chooses the measurement basis randomly by choosing which PBS subnetwork to
    # use, one of which includes a half-wave plate in the path before the PBS
    bs = aqnsim.BeamSplitter(env=node.env, qs=qs, name="bs")
    node.add_subcomponent(bs)
    node.ports["entangled_photon_input_port"].forward_input_to_input(bs.ports[aqnsim.BS_LEFT_INPUT_PORT_NAME])

    # create a polarizing beamsplitter and two detectors for each output port
    # of the input beamsplitter
    bs_outport_names = [aqnsim.BS_LEFT_OUTPUT_PORT_NAME, aqnsim.BS_RIGHT_OUTPUT_PORT_NAME]
    pbs_outport_names = [aqnsim.OUTPUT_PORT_0_NAME, aqnsim.OUTPUT_PORT_1_NAME]
    for n in range(2):
        # add polarizing beamsplitter (PBS)
        pbs = aqnsim.PolarizingBeamSplitter(env=node.env, qs=qs, name=f"pbs{n}")
        node.add_subcomponent(pbs)

        # add a half-wave plate on one of the output ports of the input beamsplitter
        # note: the choice of angle = pi/8, which corresponds to theta = pi/4 on the
        # Bloch sphere, and phase phi = pi for the HWP will transform |H> to |+>
        # and |V> to |->, i.e. a Z to X basis transformation
        if n == 1:
            hwp = aqnsim.HalfWavePlate(env=node.env, qs=qs, angle=np.pi / 8, name="hwp")
            bs.ports[bs_outport_names[n]].forward_output_to_input(hwp.ports[aqnsim.INPUT_PORT_0_NAME])
            hwp.ports[aqnsim.OUTPUT_PORT_0_NAME].forward_output_to_input(pbs.ports[aqnsim.INPUT_PORT_0_NAME])
            node.add_subcomponent(hwp)
        else:
            bs.ports[bs_outport_names[n]].forward_output_to_input(pbs.ports[aqnsim.INPUT_PORT_0_NAME])

        # add photon detectors
        for k in range(2):
            detector = aqnsim.PhotonDetector(
                env=node.env,
                qs=qs,
                number_resolving=False,
                error_on_fail=False,
                delete_measured_qubit=False,
                dark_count_rates=DETECTOR_DARK_COUNTS.copy(),
                downtime=DETECTOR_DEAD_TIME,
                efficiency_params=DETECTOR_EFFICIENCY,
                name=f"{node.name}_detector_{n}_{k}",
            )
            node.add_subcomponent(detector)
            pbs.ports[pbs_outport_names[k]].forward_output_to_input(detector.ports[aqnsim.PD_INPUT_PORT_NAME])


"""
    Run the BBM92 example
"""


def compute_QBER(node0_key: list, node1_key: list):
    """Estimate QBER between two nodes receiving entangled photons.

    :param node0_key: the first node's sifted key; a list of 0 and 1
    :param node1_key: the second node's sifted key; a list of 0 and 1
    """
    N = len(node0_key)
    if N == 0:
        qber = 1
    else:
        qber = 1 - (sum(1 if node0_key[i] == node1_key[i] else 0 for i in range(N)) / N)
    return qber


def run_example(time: float):
    """Runs the simulation
    :param time: The amount of time to run the simulation for
    """
    # instantiate environment and QuantumSimulator
    env = simpy.Environment()
    qs = aqnsim.QuantumSimulator()
    aqnsim.simlogger.configure(env=env)
    aqnsim.simlogger.info("creating simulation environment and quantum simulator")

    # setup network and protocols, and run sim until the given time
    aqnsim.simlogger.info("setting up network topology")
    network = setup_network(env=env, qs=qs)

    # run the simulation
    env.run(until=time)

    # post-process the results
    receiving_node0 = network.subcomponents["Alice"]
    receiving_node1 = network.subcomponents["Bob"]
    receiving_node0_protocol = receiving_node0.protocols["ReceiveProtocol"]
    receiving_node1_protocol = receiving_node1.protocols["ReceiveProtocol"]
    aqnsim.simlogger.info(f"Alice sifted key: {receiving_node0_protocol.sifted_key}")
    aqnsim.simlogger.info(f"Bob sifted key: {receiving_node1_protocol.sifted_key}")

    # Compute QBER
    qber = compute_QBER(receiving_node0_protocol.sifted_key, receiving_node1_protocol.sifted_key)
    aqnsim.simlogger.info(f"QBER of keys: {qber}")

    # Estimate sifted key rate
    sifted_key_rate = len(receiving_node0_protocol.sifted_key) / COLLECTION_TIME
    secure_key_rate = max(sifted_key_rate * (1 - 2.1 * aqnsim.binary_entropy(qber)), 0)
    aqnsim.simlogger.info(f"Secure key rate (Hz): {secure_key_rate}")

    aqnsim.simlogger.info("processing complete.")


if __name__ == "__main__":
    run_example(time=SIMULATION_TIME)
