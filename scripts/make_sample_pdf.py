"""Generate data/sample_bluetooth_primer.pdf, a small offline stand-in for the spec.

The official Bluetooth Core Specification is copyrighted by the Bluetooth SIG and
too large to ship in a repository, so the app falls back to this self-written
primer (~6 pages) when the real spec is not available. It is a *teaching*
document summarising well-known concepts; always verify details against the
official specification at https://www.bluetooth.com/specifications/specs/.

Usage:
    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_bluetooth_primer.pdf"

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "1 Introduction and Scope",
        [
            "This primer is an educational summary of selected topics from the Bluetooth Core Specification. "
            "It is NOT the official specification. It exists so that a Retrieval-Augmented Generation (RAG) "
            "demo can run without downloading the full specification, which is several thousand pages long.",
            "Bluetooth is a short-range wireless technology operating in the unlicensed 2.4 GHz Industrial, "
            "Scientific and Medical (ISM) band, from 2400 MHz to 2483.5 MHz. The Core Specification defines two "
            "main radio technologies: Basic Rate / Enhanced Data Rate (BR/EDR), often called Classic Bluetooth, "
            "and Bluetooth Low Energy (LE). A device that implements both is called a dual-mode device.",
            "The architecture is split into a Host and a Controller. The Controller contains the radio, the "
            "baseband or link layer, and the link manager. The Host contains upper layers such as L2CAP, the "
            "Attribute Protocol (ATT), the Generic Attribute Profile (GATT), the Security Manager Protocol (SMP) "
            "and the Generic Access Profile (GAP). Host and Controller communicate through the Host Controller "
            "Interface (HCI), which defines commands, events and data packets over a transport such as UART or USB.",
        ],
    ),
    (
        "2 BR/EDR Radio",
        [
            "The BR/EDR radio uses 79 RF channels spaced 1 MHz apart, from 2402 MHz to 2480 MHz. Channel k has "
            "the centre frequency f = 2402 + k MHz, for k = 0 to 78.",
            "Basic Rate uses Gaussian Frequency Shift Keying (GFSK) with a symbol rate of 1 Msym/s, giving a raw "
            "bit rate of 1 Mb/s. Enhanced Data Rate keeps the GFSK access code and header but switches the payload "
            "to phase shift keying: pi/4-DQPSK provides 2 Mb/s and 8DPSK provides 3 Mb/s.",
            "A BR/EDR piconet consists of one Central (historically called master) and up to seven active "
            "Peripherals (historically called slaves). The Central defines the piconet clock and the frequency "
            "hopping sequence. In the connection state the radio hops at a nominal rate of 1600 hops per second, "
            "one hop per 625 microsecond time slot; multi-slot packets occupy 3 or 5 slots on a single frequency.",
        ],
    ),
    (
        "3 Transmitter Power Classes",
        [
            "Devices are classified by the maximum output power they can transmit. For BR/EDR the power classes are:",
            "Power Class 1: maximum output power 100 mW (+20 dBm), minimum 1 mW (0 dBm). Power control is "
            "mandatory for Class 1 devices operating above +4 dBm.",
            "Power Class 2: maximum output power 2.5 mW (+4 dBm), minimum 0.25 mW (-6 dBm).",
            "Power Class 3: maximum output power 1 mW (0 dBm).",
            "For Bluetooth Low Energy the specification defines the following LE power classes, all with a "
            "minimum output power of 0.01 mW (-20 dBm):",
            "LE Power Class 1: maximum output power 100 mW (+20 dBm).",
            "LE Power Class 1.5: maximum output power 10 mW (+10 dBm).",
            "LE Power Class 2: maximum output power 2.5 mW (+4 dBm).",
            "LE Power Class 3: maximum output power 1 mW (0 dBm).",
            "Higher transmit power increases range but also increases current consumption and interference. "
            "Many LE devices therefore use transmit power control, adjusting the output power according to the "
            "received signal strength (RSSI) reported by the peer.",
        ],
    ),
    (
        "4 Adaptive Frequency Hopping (AFH) in BR/EDR",
        [
            "Adaptive Frequency Hopping was introduced in version 1.2 of the specification to improve coexistence "
            "with other systems in the 2.4 GHz band, most notably Wi-Fi. Instead of hopping over all 79 channels, "
            "a piconet using AFH hops only over channels classified as 'used', avoiding channels that suffer from "
            "interference.",
            "AFH works in three steps. First, channel classification: each channel is classified as used, unused "
            "or unknown. The Central may measure channel quality itself, may request classification reports "
            "from Peripherals, and the Host may provide its own information with the HCI command "
            "HCI_Set_AFH_Host_Channel_Classification. Second, the Central communicates the resulting AFH channel "
            "map to the Peripherals with the LMP_SET_AFH message, which includes the instant at which the new map "
            "takes effect so that all devices switch simultaneously. Third, the hop selection kernel remaps any hop "
            "that would land on an unused channel onto one of the used channels.",
            "The specification requires that at least N_min = 20 channels remain in the used set. This keeps the "
            "system a frequency-hopping spread spectrum system for regulatory purposes and preserves robustness. "
            "Channel classification is dynamic: as interference moves, the Central can update the map.",
            "AFH does not change the hop rate. Packets on the same slot pair still follow the same-channel "
            "mechanism, where the Peripheral responds on the same frequency used by the Central in the preceding slot.",
        ],
    ),
    (
        "5 Bluetooth Low Energy Radio and PHYs",
        [
            "The LE radio uses 40 RF channels spaced 2 MHz apart, from 2402 MHz to 2480 MHz. Three of them are "
            "primary advertising channels: channel index 37 (2402 MHz), 38 (2426 MHz) and 39 (2480 MHz). They are "
            "placed to avoid the most commonly used Wi-Fi channels 1, 6 and 11. The remaining 37 channels "
            "(indices 0 to 36) are general purpose channels used for connections and secondary advertising.",
            "LE defines three PHYs. LE 1M uses GFSK at 1 Msym/s and is mandatory. LE 2M, introduced in version 5.0, "
            "doubles the symbol rate to 2 Msym/s for higher throughput and shorter air time. LE Coded, also from "
            "version 5.0, uses the 1 Msym/s modulation with forward error correction: coding S=2 yields 500 kb/s and "
            "coding S=8 yields 125 kb/s, trading data rate for roughly two to four times the range.",
        ],
    ),
    (
        "6 LE Link Layer States and Connections",
        [
            "The LE Link Layer is described as a state machine. The basic states are Standby, Advertising, "
            "Scanning, Initiating and Connection. Later versions add the Synchronization state for periodic "
            "advertising and Isochronous Broadcasting states for LE Audio.",
            "An advertiser transmits advertising packets on the primary advertising channels at an advertising "
            "interval between 20 ms and 10.24 s. A random delay advDelay of 0 to 10 ms is added to every advertising "
            "event so that two advertisers do not collide repeatedly.",
            "A connection is created when an initiator responds to a connectable advertisement with a connection "
            "request. Afterwards data exchange is organised into connection events. The connection interval is a "
            "multiple of 1.25 ms between 7.5 ms and 4.0 s. The Peripheral latency parameter allows the Peripheral "
            "to skip a number of consecutive connection events to save power when it has no data to send. The "
            "supervision timeout, between 100 ms and 32 s, is the maximum time between two received packets "
            "before the link is considered lost.",
        ],
    ),
    (
        "7 LE Channel Selection and Channel Maps",
        [
            "LE connections also hop between channels, once per connection event. Channel Selection Algorithm #1 "
            "uses a fixed hop increment between 5 and 16: unmappedChannel = (lastUnmappedChannel + hopIncrement) "
            "mod 37. If the unmapped channel is marked unused in the channel map, it is remapped onto the table of "
            "used channels.",
            "Channel Selection Algorithm #2, introduced in version 5.0, derives a pseudo-random channel from the "
            "connection's Access Address and the connection event counter. It spreads traffic more evenly and "
            "is also used for periodic advertising and isochronous channels.",
            "The Central maintains the channel map, classifying each of the 37 data channels as used or unused, and "
            "updates it with the LL_CHANNEL_MAP_IND control PDU, which contains the new map and the instant at "
            "which it applies. The Host can contribute with HCI_LE_Set_Host_Channel_Classification. In LE, a "
            "channel map shall contain at least two used channels. This is the LE equivalent of AFH.",
        ],
    ),
    (
        "8 Attribute Protocol and GATT",
        [
            "The Attribute Protocol (ATT) lets a client read and write attributes stored on a server. Each attribute "
            "has a 16-bit handle, a type identified by a UUID, a value and permissions. The default ATT_MTU on LE "
            "is 23 bytes; client and server can negotiate a larger MTU with the Exchange MTU procedure.",
            "The Generic Attribute Profile (GATT) organises attributes into a hierarchy of services, characteristics "
            "and descriptors. A characteristic declaration is followed by its value attribute and optional "
            "descriptors such as the Client Characteristic Configuration Descriptor (CCCD), which a client writes "
            "to enable notifications or indications. Notifications are unacknowledged, while indications require "
            "a confirmation from the client.",
        ],
    ),
    (
        "9 Security: Pairing and Bonding",
        [
            "In LE, pairing is performed by the Security Manager Protocol (SMP). LE Legacy Pairing is the original "
            "method. LE Secure Connections, introduced in version 4.2, uses Elliptic Curve Diffie-Hellman (ECDH) "
            "on the P-256 curve to derive the Long Term Key (LTK), protecting against passive eavesdropping.",
            "The association model is chosen from the devices' input and output capabilities: Just Works (no "
            "user interaction and no protection against man-in-the-middle attacks), Numeric Comparison (both "
            "devices display a six-digit number the user confirms; Secure Connections only), Passkey Entry (a "
            "six-digit passkey is typed on one or both devices) and Out of Band (keys are exchanged over another "
            "channel such as NFC).",
            "Bonding means storing the keys produced by pairing so that future connections can be encrypted "
            "without pairing again. LE link encryption uses AES-CCM with a 128-bit key.",
        ],
    ),
    (
        "10 Glossary",
        [
            "AFH - Adaptive Frequency Hopping. ATT - Attribute Protocol. BR/EDR - Basic Rate / Enhanced Data Rate. "
            "CCCD - Client Characteristic Configuration Descriptor. GAP - Generic Access Profile. GATT - Generic "
            "Attribute Profile. HCI - Host Controller Interface. ISM - Industrial, Scientific and Medical band. "
            "L2CAP - Logical Link Control and Adaptation Protocol. LE - Low Energy. LMP - Link Manager Protocol. "
            "LTK - Long Term Key. MTU - Maximum Transmission Unit. PHY - Physical layer. RSSI - Received Signal "
            "Strength Indicator. SMP - Security Manager Protocol.",
        ],
    ),
]


class PrimerPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, "Bluetooth Primer for RAG Experiments - educational sample, NOT the official specification", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


def main() -> None:
    pdf = PrimerPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(22, 18, 22)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, "Bluetooth Primer for RAG Experiments", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, "A compact, self-written summary of Bluetooth Core Specification topics used as a fallback document.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for i, (title, paragraphs) in enumerate(SECTIONS):
        if i in (3, 6, 8):  # spread sections over several pages so page citations are meaningful
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 10.5)
        for para in paragraphs:
            pdf.multi_cell(0, 5.6, para, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2.5)
        pdf.ln(3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Wrote {OUT} ({pdf.page_no()} pages)")


if __name__ == "__main__":
    main()
