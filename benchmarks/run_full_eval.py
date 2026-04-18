#!/usr/bin/env python3
"""Comprehensive synthetic benchmark for athena-verify.

Creates 100+ realistic test cases across 6 hallucination categories,
runs athena_verify.verify() with multiple NLI models and trust thresholds,
and reports per-category and overall precision, recall, F1, ECE, and latency.

Usage:
    python benchmarks/run_full_eval.py
    python benchmarks/run_full_eval.py --output benchmarks/results/full_eval.json
    python benchmarks/run_full_eval.py --thresholds 0.50 0.60 0.70 0.80
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from athena_verify import verify


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def compute_ece(
    predicted_scores: list[float],
    actual_labels: list[bool],
    n_bins: int = 10,
) -> float:
    if not predicted_scores:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total = len(predicted_scores)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        indices = [
            j
            for j, s in enumerate(predicted_scores)
            if lo <= s < hi or (i == n_bins - 1 and s == hi)
        ]
        if not indices:
            continue
        bin_acc = sum(1 for j in indices if actual_labels[j]) / len(indices)
        bin_conf = sum(predicted_scores[j] for j in indices) / len(indices)
        ece += len(indices) / total * abs(bin_acc - bin_conf)

    return ece


def build_faithful_cases() -> list[dict]:
    return [
        {
            "category": "faithful",
            "domain": "legal",
            "question": "What is the indemnification cap in this agreement?",
            "answer": "The indemnification cap is set at $2 million per incident. This cap applies to all claims arising under Section 12. The cap does not apply to breaches of confidentiality obligations.",
            "context": [
                "INDEMNIFICATION. The Indemnifying Party shall indemnify the Indemnified Party against all losses up to $2 million per incident arising under Section 12 of this Agreement.",
                "Section 14.3: The indemnification caps stated in Section 12 shall not apply to breaches of the confidentiality provisions set forth in Section 8.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "legal",
            "question": "What are the termination provisions?",
            "answer": "Either party may terminate this agreement with 90 days written notice. Termination does not relieve either party of obligations accrued prior to the termination date. All outstanding invoices must be paid within 30 days of termination.",
            "context": [
                "TERMINATION. Either Party may terminate this Agreement at any time by providing ninety (90) days prior written notice to the other Party.",
                "Effect of Termination. Termination shall not relieve either Party of any obligations accruing prior to the effective date of termination.",
                "Upon termination, all outstanding invoices shall become due and payable within thirty (30) days of the effective termination date.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "legal",
            "question": "What governing law applies to this contract?",
            "answer": "This agreement is governed by the laws of the State of Delaware, without regard to its conflict of law principles. Any disputes shall be resolved in the Court of Chancery of Delaware.",
            "context": [
                "GOVERNING LAW. This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without giving effect to any choice or conflict of law provision or rule.",
                "DISPUTE RESOLUTION. Any dispute arising under or in connection with this Agreement shall be subject to the exclusive jurisdiction of the Court of Chancery of the State of Delaware.",
            ],
            "sentence_labels": ["supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "legal",
            "question": "What intellectual property rights are retained by the licensor?",
            "answer": "The licensor retains all right, title, and interest in and to the licensed software. The licensee receives a non-exclusive, non-transferable license to use the software. No ownership rights are transferred under this agreement.",
            "context": [
                "OWNERSHIP. Licensor shall retain all right, title, and interest in and to the Licensed Software, including all intellectual property rights therein.",
                "LICENSE GRANT. Subject to the terms of this Agreement, Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Licensed Software.",
                "No ownership rights in the Licensed Software are transferred to Licensee under this Agreement.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "medical",
            "question": "What is the recommended first-line treatment for community-acquired pneumonia in healthy adults?",
            "answer": "For healthy adults under 65 with community-acquired pneumonia, amoxicillin is the recommended first-line treatment. The standard dosage is 500 mg three times daily for 5 days. Patients should be reassessed if symptoms do not improve within 48 hours.",
            "context": [
                "Community-Acquired Pneumonia (CAP) in Adults: For immunocompetent adults under 65 years without comorbidities, first-line therapy is amoxicillin 500 mg three times daily (TDS) for 5 days.",
                "Clinical Review: Patients who do not show symptomatic improvement within 48 hours of initiating antibiotic therapy should be reassessed for complications or alternative diagnoses.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "medical",
            "question": "What are the contraindications for metformin?",
            "answer": "Metformin is contraindicated in patients with an eGFR below 30 mL/min. It should also be avoided in patients with metabolic acidosis. The medication should be temporarily discontinued prior to iodinated contrast procedures.",
            "context": [
                "CONTRAINDICATIONS: Metformin is contraindicated in patients with an estimated glomerular filtration rate (eGFR) less than 30 mL/min/1.73 m².",
                "Metformin is contraindicated in patients with known hypersensitivity to metformin hydrochloride or with metabolic acidosis, including diabetic ketoacidosis.",
                "Before administering iodinated contrast, metformin should be temporarily discontinued at the time of or prior to the procedure.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "medical",
            "question": "What is the recommended vaccination schedule for infants?",
            "answer": "The DTaP vaccine should be administered at 2, 4, and 6 months of age. The first dose of MMR vaccine is given at 12-15 months. The varicella vaccine is administered as a two-dose series starting at 12-15 months.",
            "context": [
                "DTaP (Diphtheria, Tetanus, and Pertussis): Administer to children at ages 2, 4, and 6 months, with additional doses at 15-18 months and 4-6 years.",
                "MMR (Measles, Mumps, Rubella): The first dose is recommended at 12-15 months of age, with a second dose at 4-6 years.",
                "Varicella (Chickenpox): A two-dose series is recommended, with the first dose administered at 12-15 months and the second dose at 4-6 years of age.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "medical",
            "question": "What is the blood pressure target for adults with diabetes?",
            "answer": "The target blood pressure for adults with diabetes is below 140/90 mmHg. A lower target of 130/80 may be considered for younger patients. Lifestyle modifications should be initiated alongside pharmacotherapy.",
            "context": [
                "In adults with diabetes mellitus, the recommended blood pressure target is <140/90 mmHg.",
                "A lower target of <130/80 mmHg may be considered in younger patients with diabetes if it can be achieved without undue treatment burden.",
                "Lifestyle modifications including dietary sodium reduction and increased physical activity should be initiated concurrently with antihypertensive pharmacotherapy in diabetic patients.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "technical",
            "question": "How do you configure a Redis connection pool?",
            "answer": "To configure a Redis connection pool, use the ConnectionPool class from redis-py. Set the max_connections parameter to control the pool size. The default connection timeout is 2 seconds.",
            "context": [
                "Redis-py ConnectionPool: The ConnectionPool class manages a set of connections to a Redis server. Use redis.ConnectionPool(max_connections=N) to set the maximum number of connections.",
                "By default, connections in the pool have a 2-second connect timeout. This can be overridden via the socket_connect_timeout parameter.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "technical",
            "question": "What are the system requirements for running Kubernetes?",
            "answer": "Kubernetes requires a minimum of 2 CPUs for the control plane node. At least 2 GB of RAM is needed for the control plane. The container runtime must be compatible with the CRI specification.",
            "context": [
                "Kubernetes Control Plane Requirements: Minimum 2 CPUs, 2 GB RAM. A container runtime compatible with the Kubernetes CRI (Container Runtime Interface) must be installed on all nodes.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "technical",
            "question": "How does TLS 1.3 handshake differ from TLS 1.2?",
            "answer": "TLS 1.3 reduces the handshake to one round trip compared to two in TLS 1.2. It also eliminates support for weak cipher suites like RC4 and 3DES. The new handshake supports 0-RTT data for resumed sessions.",
            "context": [
                "TLS 1.3 improves upon TLS 1.2 by reducing the handshake from two round trips to one. Cipher suites such as RC4, 3DES, and AES-CBC have been removed.",
                "TLS 1.3 introduces 0-RTT (zero round-trip time) resumption, allowing clients to send application data immediately when resuming a session.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "technical",
            "question": "What is the default garbage collector in Python 3.12?",
            "answer": "Python 3.12 uses a generational garbage collector with three generations. The GC uses reference counting as the primary memory management mechanism. Cyclic garbage collection handles reference cycles.",
            "context": [
                "Python's memory management uses reference counting as the primary mechanism, supplemented by a generational cyclic garbage collector with three generations (0, 1, and 2).",
                "In Python 3.12, the default garbage collector continues to use the three-generation model to detect and collect cyclic references that reference counting alone cannot reclaim.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "general",
            "question": "What are the opening hours of the Louvre Museum?",
            "answer": "The Louvre Museum is open daily from 9 AM to 6 PM, except on Tuesdays when it is closed. On Wednesdays and Fridays, the museum stays open until 9:45 PM. Admission is free on the first Sunday of each month from October to March.",
            "context": [
                "Louvre Museum Hours: Open every day from 9:00 AM to 6:00 PM, except Tuesdays (closed). Late openings until 9:45 PM on Wednesdays and Fridays.",
                "Free admission is offered on the first Sunday of each month from October through March.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "general",
            "question": "What is the process for obtaining a US passport?",
            "answer": "You must submit Form DS-11 in person at a passport acceptance facility. Proof of US citizenship such as a birth certificate is required. Standard processing time is 6-8 weeks.",
            "context": [
                "To apply for a new US passport, complete Form DS-11 and submit it in person at a passport acceptance facility.",
                "Required documents include proof of US citizenship (such as a certified birth certificate) and a valid government-issued photo ID.",
                "Current processing times: Standard service is 6-8 weeks. Expedited service is 2-3 weeks for an additional fee.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "general",
            "question": "What ingredients are needed for a classic margherita pizza?",
            "answer": "A classic margherita pizza requires pizza dough, San Marzano tomato sauce, fresh mozzarella, and fresh basil. Olive oil is drizzled on top before baking. The pizza is traditionally baked at 450°C for 90 seconds.",
            "context": [
                "Classic Margherita Pizza Ingredients: Pizza dough, San Marzano tomato sauce, fresh mozzarella cheese, fresh basil leaves, extra-virgin olive oil.",
                "Traditionally, the pizza is baked in a wood-fired oven at approximately 450°C (842°F) for about 60-90 seconds.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "legal",
            "question": "What is the liability limit under this SLA?",
            "answer": "The total aggregate liability under this SLA is capped at 12 months of fees. This limitation does not apply to data breaches caused by gross negligence. Each claim is subject to a deductible equal to one month of fees.",
            "context": [
                "LIMITATION OF LIABILITY. The total aggregate liability of the Service Provider under this Service Level Agreement shall not exceed twelve (12) months of service fees paid by the Customer in the twelve months preceding the claim.",
                "EXCEPTIONS. The foregoing limitation shall not apply to claims arising from data breaches caused by the gross negligence or willful misconduct of the Service Provider.",
                "Each claim submitted under this SLA is subject to a deductible equal to one (1) month of service fees.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "medical",
            "question": "What is the recommended dosage of amoxicillin for acute otitis media in children?",
            "answer": "The recommended dosage of amoxicillin for acute otitis media is 80-90 mg/kg/day divided into two doses. Treatment duration is typically 10 days for children under 2 years. For children over 2 years with mild symptoms, 5-7 days may be sufficient.",
            "context": [
                "Acute Otitis Media — Treatment: Amoxicillin 80-90 mg/kg/day in 2 divided doses is the first-line treatment.",
                "Duration: Standard treatment is 10 days for children younger than 2 years. For children 2 years and older with mild-moderate disease, a 5-7 day course may be appropriate.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "technical",
            "question": "How do you enable HTTPS on nginx?",
            "answer": "To enable HTTPS on nginx, you need to configure the server block with the SSL certificate and key paths. Set listen directive to port 443 with ssl parameter. Use ssl_protocols to specify TLS 1.2 and TLS 1.3.",
            "context": [
                "To configure HTTPS in nginx, add an SSL certificate and key to your server block configuration using the ssl_certificate and ssl_certificate_key directives.",
                "Set the listen directive to 'listen 443 ssl;' to enable SSL on port 443.",
                "Recommended: Use 'ssl_protocols TLSv1.2 TLSv1.3;' to restrict to modern TLS versions.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "general",
            "question": "What is the carbon tax rate in Sweden?",
            "answer": "Sweden has a carbon tax of approximately 1200 SEK per tonne of CO2. It was introduced in 1991 as one of the first carbon taxes in the world. Industries covered by the EU ETS are exempt from the national carbon tax.",
            "context": [
                "Sweden's carbon tax, introduced in 1991, is among the highest in the world at approximately SEK 1,200 per tonne of CO2.",
                "Industries that are covered by the EU Emissions Trading System (EU ETS) are exempt from the Swedish carbon tax.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "legal",
            "question": "What non-compete restrictions apply after employment?",
            "answer": "The employee agrees not to compete with the company for 12 months following termination. The non-compete applies within a 50-mile radius of any company office. The employee must not solicit any company clients for 18 months after leaving.",
            "context": [
                "NON-COMPETE. Employee agrees that for a period of twelve (12) months following termination of employment, Employee shall not engage in any competing business within a fifty (50) mile radius of any office of the Company.",
                "NON-SOLICITATION. For a period of eighteen (18) months following termination, Employee shall not solicit any client or customer of the Company.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "medical",
            "question": "What is the recommended approach for hypertensive emergency?",
            "answer": "In hypertensive emergency, the mean arterial pressure should be reduced by no more than 25% in the first hour. Intravenous labetalol or nicardipine are the preferred first-line agents. Target blood pressure should be 160/100 mmHg within 2-6 hours.",
            "context": [
                "Management of Hypertensive Emergency: Reduce mean arterial pressure (MAP) by no more than 25% within the first hour. Preferred IV agents include labetalol and nicardipine.",
                "The target blood pressure after 2-6 hours should be approximately 160/100 mmHg, with gradual further reduction over 24-48 hours.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "technical",
            "question": "How does garbage collection work in Go?",
            "answer": "Go uses a concurrent, tri-color mark-and-sweep garbage collector. The GC runs concurrently with the application to minimize pause times. Go 1.19 introduced a soft memory limit feature for the garbage collector.",
            "context": [
                "Go's garbage collector is a concurrent, tri-color mark-and-sweep collector designed to keep pause times low by running concurrently with the application.",
                "Since Go 1.19, the runtime supports a soft memory limit (GOMEMLIMIT) that allows the garbage collector to target a specified memory usage.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "general",
            "question": "What is the tallest building in the world?",
            "answer": "The Burj Khalifa in Dubai is the tallest building in the world at 828 meters. It has 163 habitable floors above ground. The building was completed in 2010 and designed by the architectural firm SOM.",
            "context": [
                "Burj Khalifa, located in Dubai, United Arab Emirates, is the world's tallest structure at 828 meters (2,717 ft). It has 163 habitable floors.",
                "Completed in January 2010, the Burj Khalifa was designed by Skidmore, Owings & Merrill (SOM).",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "legal",
            "question": "What warranty period applies to the software?",
            "answer": "The licensor warrants that the software will perform substantially in accordance with the documentation for 90 days from delivery. The sole remedy for breach of warranty is replacement or refund at the licensor's option. The licensor provides no warranty for modifications made by the licensee.",
            "context": [
                "WARRANTY. Licensor warrants that the Software will perform substantially in accordance with the accompanying documentation for a period of ninety (90) days from the date of delivery.",
                "REMEDY. Licensor's entire liability and Licensee's exclusive remedy shall be, at Licensor's option, either (a) replacement of the Software or (b) refund of the license fee.",
                "This warranty does not apply to any modifications of the Software made by Licensee.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "medical",
            "question": "What is the recommended daily sodium intake?",
            "answer": "The American Heart Association recommends limiting sodium intake to less than 2,300 mg per day. An ideal limit of no more than 1,500 mg per day is advised for most adults. Processed foods account for about 70% of dietary sodium.",
            "context": [
                "The American Heart Association (AHA) recommends no more than 2,300 mg of sodium per day, with an ideal limit of no more than 1,500 mg per day for most adults.",
                "Approximately 70% of dietary sodium consumed by Americans comes from processed, packaged, and restaurant foods.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "technical",
            "question": "What is the default replication factor in HDFS?",
            "answer": "The default replication factor in HDFS is 3. This means each block is stored on three different DataNodes. The replication factor can be configured on a per-file basis.",
            "context": [
                "HDFS Replication: By default, each block in HDFS is replicated to 3 DataNodes. The default replication factor is 3.",
                "The replication factor can be specified on a per-file basis at the time of file creation. It can also be changed for existing files.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "general",
            "question": "When is the next total solar eclipse visible from North America?",
            "answer": "A total solar eclipse will cross the United States on August 23, 2044. The path of totality will pass through several midwestern states. Another total solar eclipse occurs on August 12, 2026, visible from Greenland and Iceland.",
            "context": [
                "Upcoming Total Solar Eclipses: August 12, 2026 — visible from Greenland, Iceland, and Spain. August 23, 2044 — visible across the midwestern United States.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "legal",
            "question": "What is the data retention period under this DPA?",
            "answer": "Customer data must be retained for the duration of the agreement plus 3 years following termination. The processor must delete all customer data within 90 days of a valid deletion request. Backup copies may be retained for an additional 30-day recovery window.",
            "context": [
                "DATA RETENTION. Processor shall retain Customer Data for the term of this DPA and for three (3) years following termination or expiration.",
                "Upon receipt of a verified deletion request from Customer, Processor shall delete all Customer Data within ninety (90) days.",
                "Backup copies of Customer Data may be retained for an additional thirty (30) days for disaster recovery purposes before permanent deletion.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "medical",
            "question": "What is the recommended screening interval for colorectal cancer?",
            "answer": "Average-risk adults should begin colorectal cancer screening at age 45. Colonoscopy screening is recommended every 10 years. FIT testing can be performed annually as an alternative.",
            "context": [
                "Colorectal Cancer Screening: Adults at average risk should begin screening at age 45.",
                "Screening options include colonoscopy every 10 years, or annual fecal immunochemical test (FIT).",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
        {
            "category": "faithful",
            "domain": "technical",
            "question": "How do you configure CORS in Express.js?",
            "answer": "Use the cors middleware package to enable CORS in Express.js. The Access-Control-Allow-Origin header can be set to specific domains. Preflight requests are automatically handled when using the cors middleware.",
            "context": [
                "To enable CORS in Express.js, install the 'cors' package and use app.use(cors()). The Access-Control-Allow-Origin header can be configured to allow specific origins.",
                "The cors middleware automatically handles preflight OPTIONS requests.",
            ],
            "sentence_labels": ["supported", "supported", "supported"],
        },
    ]


def build_number_substitution_cases() -> list[dict]:
    return [
        {
            "category": "number_substitution",
            "domain": "legal",
            "question": "What is the indemnification cap?",
            "answer": "The indemnification cap is $1 million per incident. This applies to all claims under the agreement.",
            "context": [
                "The Indemnifying Party shall indemnify against all losses up to $2 million per incident arising under this Agreement.",
            ],
            "sentence_labels": ["hallucinated", "supported"],
        },
        {
            "category": "number_substitution",
            "domain": "legal",
            "question": "What is the notice period for termination?",
            "answer": "Either party may terminate with 30 days written notice. Termination becomes effective upon delivery of the notice.",
            "context": [
                "Either Party may terminate this Agreement by providing ninety (90) days prior written notice. Termination shall become effective upon receipt of such notice by the other Party.",
            ],
            "sentence_labels": ["hallucinated", "supported"],
        },
        {
            "category": "number_substitution",
            "domain": "legal",
            "question": "What is the late payment penalty?",
            "answer": "Late payments incur a penalty of 1.5% per month. The maximum penalty is capped at 10% of the outstanding balance.",
            "context": [
                "Late Payment: Overdue amounts shall accrue interest at the rate of 2% per month, up to a maximum of 18% of the outstanding balance.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated"],
        },
        {
            "category": "number_substitution",
            "domain": "medical",
            "question": "What is the recommended dose of lisinopril for hypertension?",
            "answer": "The starting dose of lisinopril is 5 mg once daily. The maximum recommended dose is 20 mg per day.",
            "context": [
                "Lisinopril for Hypertension: Starting dose 10 mg once daily. May be titrated up to a maximum of 40 mg once daily.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated"],
        },
        {
            "category": "number_substitution",
            "domain": "medical",
            "question": "How long is the quarantine period for COVID-19?",
            "answer": "The recommended isolation period is 14 days from symptom onset. A negative test is required on day 10 to end isolation early.",
            "context": [
                "COVID-19 Isolation Guidance: Individuals should isolate for 5 days from symptom onset. If symptoms are resolving, isolation may end after day 5 with continued masking through day 10.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated"],
        },
        {
            "category": "number_substitution",
            "domain": "medical",
            "question": "What is the gestational age for full-term birth?",
            "answer": "A full-term pregnancy is defined as 37 weeks of gestation. Preterm birth occurs before 35 weeks.",
            "context": [
                "Full-term birth is defined as delivery between 39 weeks 0 days and 40 weeks 6 days of gestation. Early term is 37 weeks 0 days through 38 weeks 6 days. Preterm birth is defined as delivery before 37 weeks 0 days.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated"],
        },
        {
            "category": "number_substitution",
            "domain": "technical",
            "question": "What is the default connection pool size in PostgreSQL?",
            "answer": "The default maximum connections in PostgreSQL is 50. This can be increased by modifying the max_connections parameter.",
            "context": [
                "PostgreSQL Configuration: The default max_connections setting is 100. This parameter can only be set at server start.",
            ],
            "sentence_labels": ["hallucinated", "supported"],
        },
        {
            "category": "number_substitution",
            "domain": "technical",
            "question": "What is the maximum payload size for AWS Lambda?",
            "answer": "The maximum payload size for synchronous Lambda invocations is 1 MB. Asynchronous invocations have a limit of 128 KB.",
            "context": [
                "AWS Lambda Limits: Synchronous invocation payload: 6 MB. Asynchronous invocation payload: 256 KB.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated"],
        },
        {
            "category": "number_substitution",
            "domain": "technical",
            "question": "How long is the TTL for DNS A records by default?",
            "answer": "The default TTL for DNS A records is 3600 seconds. Some DNS providers set it to 1800 seconds by default.",
            "context": [
                "DNS Default TTL Values: The most common default TTL for DNS A records is 86400 seconds (24 hours). Some providers default to 14400 seconds (4 hours).",
            ],
            "sentence_labels": ["hallucinated", "hallucinated"],
        },
        {
            "category": "number_substitution",
            "domain": "general",
            "question": "What is the speed limit on German autobahns?",
            "answer": "The recommended speed on German autobahns is 100 km/h. There is a hard speed limit of 200 km/h in unrestricted zones.",
            "context": [
                "German Autobahn Speed Limits: While much of the autobahn network has no general speed limit, a recommended speed (Richtgeschwindigkeit) of 130 km/h applies. There is no hard maximum speed limit on unrestricted sections.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated"],
        },
        {
            "category": "number_substitution",
            "domain": "legal",
            "question": "What is the warranty period for the equipment?",
            "answer": "The equipment warranty covers defects for 12 months from installation. Extended warranty is available for an additional 3 years.",
            "context": [
                "EQUIPMENT WARRANTY: All equipment is warranted against defects in materials and workmanship for twenty-four (24) months from the date of installation. An extended warranty plan of up to five (5) additional years is available for purchase.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated"],
        },
        {
            "category": "number_substitution",
            "domain": "medical",
            "question": "What is the survival rate for stage 3 breast cancer?",
            "answer": "The 5-year relative survival rate for stage 3 breast cancer is approximately 60%. Early detection improves the survival rate significantly.",
            "context": [
                "Breast Cancer Survival Rates: The 5-year relative survival rate for regional stage (stage III) breast cancer is approximately 86%. Early detection through screening significantly improves outcomes.",
            ],
            "sentence_labels": ["hallucinated", "supported"],
        },
        {
            "category": "number_substitution",
            "domain": "technical",
            "question": "What is the maximum message size in Apache Kafka?",
            "answer": "The default maximum message size in Kafka is 512 KB. This can be increased by setting the message.max.bytes configuration.",
            "context": [
                "Apache Kafka Configuration: The default value of message.max.bytes is 1,048,588 bytes (approximately 1 MB). This can be increased via broker configuration.",
            ],
            "sentence_labels": ["hallucinated", "supported"],
        },
        {
            "category": "number_substitution",
            "domain": "general",
            "question": "How deep is the Mariana Trench?",
            "answer": "The Mariana Trench reaches a maximum depth of approximately 8,000 meters. It is located in the western Pacific Ocean.",
            "context": [
                "The Mariana Trench, located in the western Pacific Ocean, is the deepest oceanic trench on Earth, reaching a maximum depth of approximately 10,935 meters (35,876 ft) at the Challenger Deep.",
            ],
            "sentence_labels": ["hallucinated", "supported"],
        },
        {
            "category": "number_substitution",
            "domain": "legal",
            "question": "What is the early termination fee?",
            "answer": "The early termination fee is $500 if canceled within the first 6 months. After 6 months, the fee is reduced to $100.",
            "context": [
                "EARLY TERMINATION FEE: If the Customer terminates this Agreement within the first twelve (12) months, a termination fee of $2,500 shall apply. After twelve months, the fee is reduced to $750.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated"],
        },
    ]


def build_fabricated_claims_cases() -> list[dict]:
    return [
        {
            "category": "fabricated",
            "domain": "legal",
            "question": "What are the payment terms?",
            "answer": "Payment is due within 45 days of invoice date. A 3% early payment discount is available. Invoices must be submitted through the SAP payment portal. Late payments accrue interest at the SOFR rate plus 400 basis points.",
            "context": [
                "INVOICING. The Service Provider shall submit invoices on a monthly basis. Payment is due net thirty (30) days from receipt of invoice.",
                "DISPUTES. Any disputed charges must be reported in writing within fifteen (15) business days of invoice receipt.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "legal",
            "question": "What insurance coverage is required?",
            "answer": "The contractor must maintain professional liability insurance of at least $10 million. Workers' compensation coverage must include an employer's liability limit of $1 million per accident. All policies must name the client as an additional insured.",
            "context": [
                "INSURANCE. Contractor shall maintain commercial general liability insurance with limits of not less than $1,000,000 per occurrence.",
                "Contractor shall provide certificates of insurance to Client upon request.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "medical",
            "question": "What are the side effects of this medication?",
            "answer": "Common side effects include temporary vision loss in 15% of patients. Liver toxicity has been reported in clinical trials at a rate of 8%. The FDA has issued a black box warning for cardiac arrhythmia risk. Patients should monitor their blood sugar levels weekly.",
            "context": [
                "ADVERSE REACTIONS: The most commonly reported adverse reactions (≥5%) were nausea (12%), headache (9%), and dizziness (6%).",
                "Post-marketing reports have included rare cases of angioedema.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "medical",
            "question": "What is the recommended treatment for Stage 4 lung cancer?",
            "answer": "Immunotherapy with pembrolizumab combined with acupuncture has shown a 90% remission rate. The standard protocol requires 12 cycles of treatment over 6 months. Genetic testing for ALK mutations is mandatory before starting any treatment.",
            "context": [
                "Stage IV Non-Small Cell Lung Cancer: Treatment options include chemotherapy, immunotherapy, and targeted therapy depending on biomarker status.",
                "First-line treatment for patients with high PD-L1 expression (≥50%) may include single-agent pembrolizumab.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "technical",
            "question": "How do you configure load balancing in this setup?",
            "answer": "The load balancer uses a weighted round-robin algorithm with health checks every 10 seconds. SSL termination must be configured on port 8443 with TLS 1.0 minimum. Session affinity is maintained through cookie injection at the network layer.",
            "context": [
                "Load Balancer Configuration: Set the upstream servers in the server block. Use the least_conn directive to distribute traffic to the server with the fewest active connections.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "technical",
            "question": "What are the performance characteristics of this database?",
            "answer": "The database achieves 99.999% uptime with synchronous multi-region replication. Write throughput peaks at 2 million operations per second per node. The query optimizer uses a proprietary ML-based cost model. Automatic sharding rebalances partitions every 6 hours.",
            "context": [
                "Database Performance: The system provides high availability through leader election and automatic failover.",
                "Horizontal scaling is supported through manual sharding of collections.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "general",
            "question": "What are the visa requirements for Japan?",
            "answer": "Tourist visas for Japan require a minimum bank balance of $15,000. Processing time is exactly 21 business days. Visas are valid for multiple entries over a 10-year period. A return flight itinerary printed on airline letterhead is mandatory.",
            "context": [
                "Japan Tourist Visa: Citizens of 68 countries and regions are visa-exempt for short-term stays up to 90 days.",
                "For nationals requiring a visa, a single-entry tourist visa is typically issued for a stay of up to 15, 30, or 90 days.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "general",
            "question": "What is the refund policy?",
            "answer": "Refunds are processed within 72 hours of request. A restocking fee of 25% applies to all returns. Items must be shipped back using the prepaid FedEx label provided at purchase. Refunds are issued exclusively in store credit.",
            "context": [
                "REFUND POLICY: We accept returns within 30 days of purchase. Items must be unused and in original packaging.",
                "Refunds will be issued to the original payment method within 5-7 business days after we receive the returned item.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "legal",
            "question": "What arbitration rules apply?",
            "answer": "All disputes must be arbitrated under the rules of the London Court of International Arbitration. The arbitration panel consists of seven arbitrators appointed by the International Chamber of Commerce. Proceedings must be conducted in French. Awards are final and not subject to appeal.",
            "context": [
                "ARBITRATION. Any dispute arising under this Agreement shall be resolved by binding arbitration administered by the American Arbitration Association (AAA) under its Commercial Arbitration Rules.",
                "The arbitration shall be conducted by a single arbitrator. The proceedings shall take place in New York, New York.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "medical",
            "question": "What is the protocol for anaphylaxis treatment?",
            "answer": "Administer 5 mg of intravenous epinephrine as a first-line treatment. Follow with 100 mg of IV methylprednisolone every 15 minutes. The patient must be placed in Trendelenburg position. Intubation should be performed prophylactically in all cases.",
            "context": [
                "Anaphylaxis Management: Administer intramuscular epinephrine 0.3-0.5 mg (1:1000) into the mid-outer thigh immediately.",
                "Position the patient supine with legs elevated unless respiratory distress worsens in that position.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "technical",
            "question": "How do you set up monitoring?",
            "answer": "Metrics are collected every 500 milliseconds via a UDP broadcast protocol. Alert thresholds must be configured in XML format in the /etc/monitoring/conf.xml file. The monitoring agent requires a dedicated GPU for log processing. All dashboards are auto-generated using generative AI.",
            "context": [
                "Monitoring Setup: Install the monitoring agent on each host. Configure the agent to report metrics to the central collector.",
                "Metrics are collected at 60-second intervals by default. Set alert thresholds in the web UI under Settings > Alerts.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "general",
            "question": "What are the hotel cancellation terms?",
            "answer": "Cancellations made less than 14 days before check-in are subject to a fee equal to the entire stay cost. No-shows are charged double the room rate for each night of the reservation. Cancellations must be submitted by certified mail to the hotel's registered address.",
            "context": [
                "HOTEL CANCELLATION POLICY: Free cancellation is available up to 48 hours before check-in.",
                "Cancellations within 48 hours of check-in will be charged one night's room rate.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "legal",
            "question": "What are the confidentiality obligations?",
            "answer": "Confidential information must be encrypted using AES-256 before any transmission. Breach of confidentiality carries a statutory penalty of $50,000 per incident. All confidential documents must be printed on red paper to distinguish them from public materials.",
            "context": [
                "CONFIDENTIALITY. Receiving Party shall hold all Confidential Information in strict confidence and shall not disclose such information to any third party without prior written consent.",
                "Confidential Information shall be protected using the same degree of care used to protect the Receiving Party's own confidential information.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "medical",
            "question": "What are the dietary restrictions with this medication?",
            "answer": "Patients must completely eliminate all dairy products from their diet. Consumption of grapefruit juice must be limited to 500 mL per hour. Foods high in vitamin K must be weighed on a digital scale before eating. All meals must be consumed before 7 PM local time.",
            "context": [
                "Dietary Considerations: Take this medication with food to reduce gastrointestinal upset.",
                "Avoid consuming alcohol while taking this medication, as it may increase the risk of liver damage.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "technical",
            "question": "How do you configure the firewall rules?",
            "answer": "All firewall rules must be written in Rust and compiled to WebAssembly modules. The default deny policy applies only to IPv6 traffic. Port scanning detection triggers automatic nuclear shutdown of the data center. Rules are audited every 4 hours by an external SOC 3 compliance team.",
            "context": [
                "Firewall Configuration: Define ingress and egress rules in the security group configuration. The default policy is deny-all ingress.",
                "Rules are evaluated in order of specificity, with more specific rules taking priority.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "legal",
            "question": "What are the force majeure provisions?",
            "answer": "Force majeure events are limited exclusively to solar flares and asteroid impacts. The affected party has 12 hours to notify the other party by telegraph. Pandemic-related force majeure claims require a notarized affidavit from the Surgeon General.",
            "context": [
                "FORCE MAJEURE. Neither Party shall be liable for failure or delay in performing obligations due to events beyond its reasonable control, including but not limited to natural disasters, war, terrorism, strikes, and government actions.",
                "The affected Party shall provide notice to the other Party within five (5) business days of the force majeure event.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "medical",
            "question": "What are the MRI safety guidelines?",
            "answer": "Patients with titanium implants cannot undergo MRI under any circumstances. The MRI room must be painted blue to minimize patient anxiety during scans. Contrast agents are administered via subcutaneous injection only. The scan must be performed with the patient in a seated position.",
            "context": [
                "MRI Safety: Patients with certain implanted devices (cardiac pacemakers, cochlear implants) may be contraindicated for MRI. Consult the device manufacturer's guidelines.",
                "MRI contrast agents (gadolinium-based) are administered intravenously. Monitor the patient for allergic reactions.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "technical",
            "question": "What are the rate limiting rules?",
            "answer": "Rate limiting is enforced using a token bucket with a burst capacity of 500,000 requests. Each API key is limited to 10 requests per calendar year. Exceeding the rate limit triggers an automatic IP ban lasting 365 days. Rate limit headers are omitted from responses for security reasons.",
            "context": [
                "Rate Limiting: API requests are limited to 100 requests per minute per API key. Rate limit information is included in response headers: X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset.",
                "When the rate limit is exceeded, the API returns HTTP 429 Too Many Requests with a Retry-After header.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "general",
            "question": "What are the parking regulations?",
            "answer": "Parking is permitted only on odd-numbered days on the north side of the street. Electric vehicles are exempt from all time limits and parking fees. Violators will have their vehicles confiscated and auctioned within 24 hours. Parking meters accept only cryptocurrency payments.",
            "context": [
                "PARKING REGULATIONS: Maximum parking duration is 2 hours in all metered spaces. Payment can be made via coin, credit card, or the ParkMobile app.",
                "Overtime parking violations result in a $35 fine. Vehicles parked in fire lanes or blocking access will be towed at the owner's expense.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "fabricated",
            "domain": "general",
            "question": "What is the luggage allowance?",
            "answer": "Each passenger is allowed exactly 46.5 kg of checked baggage. Carry-on bags must be transparent and no larger than 35 cm in any dimension. Musical instruments count as a third carry-on and incur a $200 surcharge. All luggage must be wrapped in plastic film at check-in.",
            "context": [
                "BAGGAGE ALLOWANCE: Economy passengers may check one bag up to 23 kg (50 lbs). Carry-on allowance is one bag plus one personal item.",
                "Overweight baggage fees apply for bags exceeding the weight limit. Musical instruments may be carried on board if they fit in the overhead bin or under the seat.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated", "hallucinated"],
        },
    ]


def build_partial_support_cases() -> list[dict]:
    return [
        {
            "category": "partial_support",
            "domain": "legal",
            "question": "What are the payment and penalty terms?",
            "answer": "Payment is due within 30 days of invoice receipt. A late payment penalty of 5% per month applies to overdue amounts. The client must submit all invoices through the SAP portal.",
            "context": [
                "Payment Terms: All invoices are due and payable within thirty (30) days of receipt. Late payments shall accrue interest at the rate of 1.5% per month.",
            ],
            "sentence_labels": ["supported", "hallucinated", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "legal",
            "question": "What are the confidentiality requirements?",
            "answer": "Confidential information must not be disclosed to third parties without prior written consent. Breach of confidentiality results in an automatic $100,000 penalty. The receiving party must destroy all copies within 30 days of agreement termination.",
            "context": [
                "CONFIDENTIALITY. Receiving Party shall not disclose Confidential Information to any third party without prior written consent of Disclosing Party.",
                "Upon termination, Receiving Party shall return or destroy all copies of Confidential Information within thirty (30) days.",
            ],
            "sentence_labels": ["supported", "hallucinated", "supported"],
        },
        {
            "category": "partial_support",
            "domain": "medical",
            "question": "What are the dosing instructions for this antibiotic?",
            "answer": "Take amoxicillin 500 mg every 8 hours for the prescribed duration. You should consume the capsules with a full glass of grapefruit juice. Complete the entire course even if you feel better before it is finished.",
            "context": [
                "DOSING: Take amoxicillin 500 mg by mouth every 8 hours as directed by your physician. Complete the full course of therapy even if symptoms improve.",
            ],
            "sentence_labels": ["supported", "hallucinated", "supported"],
        },
        {
            "category": "partial_support",
            "domain": "medical",
            "question": "What are the post-operative care instructions?",
            "answer": "Keep the surgical incision dry for the first 48 hours. You may resume heavy lifting after 3 days. Take the prescribed pain medication only when absolutely necessary and not more than every 4 hours.",
            "context": [
                "Post-Operative Care: Keep the incision site clean and dry for the first 48 hours after surgery. Avoid heavy lifting for at least 4-6 weeks.",
                "Take prescribed pain medication as directed by your surgeon. Do not exceed the recommended dose.",
            ],
            "sentence_labels": ["supported", "hallucinated", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "technical",
            "question": "How do you deploy this application?",
            "answer": "Build the Docker image using the provided Dockerfile in the project root. Push the image to the internal ECR registry using the deploy.sh script. The application automatically scales to 1,000 containers during peak traffic.",
            "context": [
                "Deployment Guide: Build the Docker image from the Dockerfile in the project root directory. Push the image to the ECR registry using the deploy.sh script provided in the scripts/ directory.",
                "Auto-scaling is configured via the terraform modules and adjusts based on CPU utilization thresholds.",
            ],
            "sentence_labels": ["supported", "supported", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "technical",
            "question": "What are the API authentication requirements?",
            "answer": "All API requests require a Bearer token in the Authorization header. Tokens expire after 60 minutes and must be refreshed. The rate limit is 100 requests per second per token.",
            "context": [
                "Authentication: All API endpoints require a Bearer token passed in the Authorization header. Tokens are valid for 60 minutes from issuance.",
                "Use the /auth/refresh endpoint to obtain a new token before expiration. Rate limiting is applied per API key.",
            ],
            "sentence_labels": ["supported", "supported", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "general",
            "question": "What is the return policy?",
            "answer": "Items can be returned within 30 days of purchase in their original packaging. All returns are subject to a 25% restocking fee. Refunds are processed to the original payment method.",
            "context": [
                "Return Policy: We accept returns within 30 days of delivery. Items must be in their original, unopened packaging.",
                "Refunds will be credited to the original payment method within 5-7 business days.",
            ],
            "sentence_labels": ["supported", "hallucinated", "supported"],
        },
        {
            "category": "partial_support",
            "domain": "general",
            "question": "What are the membership benefits?",
            "answer": "Premium members receive free shipping on all orders. Members also get exclusive access to the VIP lounge at all airport locations worldwide. Annual membership costs $99 per year.",
            "context": [
                "Membership Benefits: Premium members enjoy free standard shipping on all domestic orders. Annual membership fee is $99 per year.",
            ],
            "sentence_labels": ["supported", "hallucinated", "supported"],
        },
        {
            "category": "partial_support",
            "domain": "legal",
            "question": "What are the audit rights?",
            "answer": "The client has the right to audit the vendor's records once per year with 30 days advance notice. Audits must be conducted during the vendor's lunar holiday period. The vendor must cover all audit-related expenses regardless of findings.",
            "context": [
                "AUDIT RIGHTS. Client may audit Vendor's records relevant to this Agreement once per calendar year, upon thirty (30) days written notice.",
                "Audits shall be conducted during normal business hours at Vendor's facilities.",
            ],
            "sentence_labels": ["supported", "hallucinated", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "medical",
            "question": "What is the recommended management for Type 2 diabetes?",
            "answer": "First-line therapy for Type 2 diabetes is metformin. Patients should aim for an HbA1c target below 6.0%. Lifestyle modifications including 300 minutes of vigorous exercise per week are recommended.",
            "context": [
                "Type 2 Diabetes Management: Metformin is recommended as first-line pharmacotherapy for Type 2 diabetes.",
                "A reasonable HbA1c target for most non-pregnant adults is <7%. Lifestyle modifications including at least 150 minutes of moderate-intensity exercise per week are recommended.",
            ],
            "sentence_labels": ["supported", "hallucinated", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "technical",
            "question": "What logging configuration is recommended?",
            "answer": "Applications should output structured JSON logs to stdout. The log level should be set to DEBUG in production environments. Logs are aggregated in Elasticsearch with a 90-day retention period.",
            "context": [
                "Logging Standards: All services must output structured JSON logs to stdout. Log levels: ERROR, WARN, INFO, DEBUG, TRACE.",
                "Production environments should use INFO as the default log level. Logs are shipped to the central Elasticsearch cluster.",
            ],
            "sentence_labels": ["supported", "hallucinated", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "general",
            "question": "What are the gym membership terms?",
            "answer": "The monthly membership fee is $49 with no annual contract required. Guests receive unlimited free access to the VIP sauna and rooftop pool. Members can freeze their membership for up to 6 months per year.",
            "context": [
                "Gym Membership: Our standard membership is $49/month with no long-term contract commitment.",
                "Members may freeze their account for up to 3 months per calendar year. Guest passes are available for $15 per visit.",
            ],
            "sentence_labels": ["supported", "hallucinated", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "legal",
            "question": "What are the IP ownership terms?",
            "answer": "All work product created by the consultant during the engagement belongs to the company. The consultant retains a 15% royalty on all revenues generated from their work. The company grants the consultant a perpetual license to use any patents filed.",
            "context": [
                "INTELLECTUAL PROPERTY. All work product, inventions, and materials created by Consultant during the term of this Agreement shall be the exclusive property of Company.",
                "Company shall have all right, title, and interest in and to such work product.",
            ],
            "sentence_labels": ["supported", "hallucinated", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "medical",
            "question": "What are the guidelines for antibiotic prophylaxis before dental procedures?",
            "answer": "Antibiotic prophylaxis is recommended for patients with prosthetic heart valves before dental procedures. A single dose of amoxicillin 2g should be taken 30 minutes before the procedure. All patients over 50 require prophylaxis regardless of cardiac status.",
            "context": [
                "Antibiotic prophylaxis prior to dental procedures is recommended for patients with prosthetic cardiac valves and certain congenital heart conditions.",
                "The recommended regimen is amoxicillin 2g taken orally 30-60 minutes before the dental procedure.",
            ],
            "sentence_labels": ["supported", "supported", "hallucinated"],
        },
        {
            "category": "partial_support",
            "domain": "technical",
            "question": "How do you set up CI/CD for this project?",
            "answer": "The project uses GitHub Actions for continuous integration. All pull requests must pass the unit test suite before merging. Deployment to production happens automatically every 15 minutes.",
            "context": [
                "CI/CD Setup: This project uses GitHub Actions for CI. All PRs must pass the test suite and linting checks before merge.",
                "Deployments to staging are triggered on merge to the main branch. Production deployments require manual approval.",
            ],
            "sentence_labels": ["supported", "supported", "hallucinated"],
        },
    ]


def build_subtle_contradiction_cases() -> list[dict]:
    return [
        {
            "category": "subtle_contradiction",
            "domain": "legal",
            "question": "Is there a non-solicitation clause?",
            "answer": "The employee is permitted to solicit company clients after termination. The non-solicitation clause was explicitly waived in Section 14. This freedom applies for a period of 24 months following departure.",
            "context": [
                "NON-SOLICITATION. Employee shall not, for a period of twenty-four (24) months following termination, directly or indirectly solicit any client or customer of the Company.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "subtle_contradiction",
            "domain": "legal",
            "question": "Is the warranty transferable?",
            "answer": "The warranty under this agreement is fully transferable to any successor entity. Assignments do not require prior written consent from the other party. The warrantor encourages secondary market transfers of warranty rights.",
            "context": [
                "ASSIGNMENT. This Agreement and the rights and obligations hereunder may not be assigned or transferred by either Party without the prior written consent of the other Party.",
                "Any attempted assignment without such consent shall be void.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "subtle_contradiction",
            "domain": "medical",
            "question": "Can this medication be taken during pregnancy?",
            "answer": "This medication is classified as Pregnancy Category A and is safe throughout pregnancy. Studies have confirmed no risk to the fetus at any dose. It may be freely recommended to all pregnant patients without restriction.",
            "context": [
                "PREGNANCY: This medication is classified as FDA Pregnancy Category X. Studies in animals or humans have demonstrated fetal abnormalities and/or there is positive evidence of human fetal risk.",
                "The drug is contraindicated in women who are or may become pregnant.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "subtle_contradiction",
            "domain": "medical",
            "question": "Is fasting required before this blood test?",
            "answer": "No fasting is required for this blood test. Patients may eat a normal meal before the blood draw. The test results are unaffected by recent food intake.",
            "context": [
                "PATIENT PREPARATION: Patient must fast for 12 hours prior to blood draw. Water is permitted. No food or beverages other than water should be consumed during the fasting period.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "subtle_contradiction",
            "domain": "technical",
            "question": "Is this service available in all regions?",
            "answer": "The service is available in all AWS regions globally. There are no geographic restrictions on deployment. Data residency requirements do not affect availability.",
            "context": [
                "SERVICE AVAILABILITY: This service is currently available in the following AWS regions only: us-east-1, us-west-2, eu-west-1, and ap-northeast-1.",
                "Data processed by this service will be stored in the region where the service is deployed and may not be replicated across regions.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "subtle_contradiction",
            "domain": "technical",
            "question": "Is the API backward compatible?",
            "answer": "All API versions are guaranteed to be backward compatible indefinitely. Existing endpoints will never be deprecated or removed. Clients do not need to update their integrations at any point.",
            "context": [
                "API VERSIONING: We maintain backward compatibility for at least 12 months after a new API version is released.",
                "Deprecated endpoints will be sunset after the 12-month deprecation period. Clients must migrate to the current API version within this period.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "subtle_contradiction",
            "domain": "legal",
            "question": "Are amendments to the agreement allowed orally?",
            "answer": "Amendments to this agreement may be made orally by mutual consent. Verbal agreements between the parties' representatives are binding. No written documentation is required for modifications.",
            "context": [
                "AMENDMENTS. No modification, amendment, or waiver of any provision of this Agreement shall be effective unless in writing and signed by both Parties.",
                "Verbal agreements or understandings not reduced to writing shall not be binding upon either Party.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "subtle_contradiction",
            "domain": "medical",
            "question": "Is this drug compatible with alcohol?",
            "answer": "This medication has no interaction with alcohol. Patients can consume alcoholic beverages freely while taking this drug. Alcohol does not affect the metabolism or efficacy of this medication.",
            "context": [
                "DRUG INTERACTIONS: Consumption of alcohol during treatment with this medication is contraindicated. Alcohol may potentiate the CNS depressant effects and increase the risk of hepatotoxicity.",
                "Patients should avoid all alcoholic beverages for the duration of treatment and for 72 hours after the last dose.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "subtle_contradiction",
            "domain": "technical",
            "question": "Is encryption required for data at rest?",
            "answer": "Encryption of data at rest is optional and left to customer discretion. The platform does not encrypt stored data by default. Data security relies entirely on network-level protections.",
            "context": [
                "DATA ENCRYPTION: All customer data at rest must be encrypted using AES-256 encryption. This is enabled by default and cannot be disabled.",
                "Encryption keys are managed through the platform's integrated key management service.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "subtle_contradiction",
            "domain": "general",
            "question": "Is registration required for this event?",
            "answer": "No registration is needed for this event. Walk-ins are welcome at any time during the event. Attendance is completely unrestricted and open to all.",
            "context": [
                "EVENT REGISTRATION: Pre-registration is mandatory. Only registered attendees with confirmed tickets will be admitted.",
                "Registration closes 48 hours before the event start time. No walk-ins will be accommodated due to venue capacity limits.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
    ]


def build_out_of_context_cases() -> list[dict]:
    return [
        {
            "category": "out_of_context",
            "domain": "legal",
            "question": "What are the terms of this employment agreement?",
            "answer": "Employees in California are entitled to paid sick leave under the Healthy Workplace Healthy Family Act. The minimum wage in California is $16 per hour as of 2024. Employers must provide meal breaks for shifts exceeding 5 hours.",
            "context": [
                "This Employment Agreement is between Acme Corporation (Employer) and the undersigned Employee. The Employee shall serve in the role of Senior Engineer.",
                "The employment is at-will and may be terminated by either party at any time with or without cause.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "out_of_context",
            "domain": "legal",
            "question": "What privacy rights do users have?",
            "answer": "Under GDPR, users have the right to data portability. The CCPA grants California residents the right to opt out of data sales. HIPAA provides patients with the right to access their medical records within 30 days.",
            "context": [
                "PRIVACY POLICY. We collect the following categories of personal information: name, email address, and usage data.",
                "We do not sell personal information to third parties. Users may request deletion of their data by contacting privacy@example.com.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "out_of_context",
            "domain": "medical",
            "question": "What is the mechanism of action of this drug?",
            "answer": "Aspirin works by irreversibly inhibiting cyclooxygenase-1 (COX-1). It also inhibits thromboxane A2 production in platelets. Low-dose aspirin is commonly used for cardiovascular prophylaxis.",
            "context": [
                "PATIENT INFORMATION LEAFLET — Drug X: Take one tablet daily with food. Do not crush or chew the tablet.",
                "If you miss a dose, take it as soon as you remember unless it is almost time for your next dose. Store at room temperature.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "out_of_context",
            "domain": "medical",
            "question": "What are the symptoms of this condition?",
            "answer": "Myocardial infarction typically presents with crushing chest pain radiating to the left arm. Diaphoresis and shortness of breath are common accompanying symptoms. Troponin levels typically rise within 4-6 hours of symptom onset.",
            "context": [
                "DISCHARGE SUMMARY: The patient was admitted on January 15 with a diagnosis of acute bronchitis. Treatment included albuterol nebulizer and oral prednisone.",
                "Patient was discharged on January 18 with improvement in symptoms. Follow-up with primary care in 1 week.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "out_of_context",
            "domain": "technical",
            "question": "How do you optimize this database query?",
            "answer": "Adding a B-tree index on the frequently filtered column can improve query performance. Using EXPLAIN ANALYZE helps identify sequential scans. Partitioning large tables by date range is a common optimization strategy.",
            "context": [
                "DATABASE SCHEMA: Table 'users' has columns: id (INT, PK), name (VARCHAR), email (VARCHAR), created_at (TIMESTAMP).",
                "Table 'orders' has columns: id (INT, PK), user_id (INT, FK), amount (DECIMAL), status (VARCHAR).",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "out_of_context",
            "domain": "technical",
            "question": "What is the recommended architecture?",
            "answer": "A microservices architecture with event-driven communication is recommended. Each service should own its own database following the database-per-service pattern. An API gateway handles authentication, rate limiting, and request routing.",
            "context": [
                "DEPLOYMENT GUIDE: The application is deployed as a single Docker container. Environment variables are configured in the .env file.",
                "The application listens on port 3000. Health check endpoint is available at /healthz.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "out_of_context",
            "domain": "general",
            "question": "What is the company's history?",
            "answer": "The company was founded in 1998 in a garage in Menlo Park. It went public in 2004 at $85 per share. The current CEO took over in 2015 and has since tripled the company's revenue.",
            "context": [
                "ABOUT US: We are a mid-sized software development company headquartered in Austin, Texas. Our team specializes in custom enterprise solutions.",
                "Contact us at info@example.com or call (512) 555-0199 for more information about our services.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "out_of_context",
            "domain": "general",
            "question": "What is the climate like in this region?",
            "answer": "The Amazon rainforest receives between 1,500 and 3,000 mm of rainfall annually. Average temperatures range from 25-28°C throughout the year. The region experiences a distinct dry season from June to November.",
            "context": [
                "TRAVEL GUIDE — Reykjavik, Iceland: Average summer temperatures are 10-13°C. Winter temperatures average -1 to 3°C.",
                "Daylight hours range from 4 hours in December to nearly 24 hours in June. Annual precipitation is approximately 800 mm.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "out_of_context",
            "domain": "legal",
            "question": "What are the export control requirements?",
            "answer": "The EAR99 classification covers low-technology consumer goods not subject to specific export controls. Items on the USML require State Department approval for export. Deemed exports occur when controlled technology is released to foreign nationals within the US.",
            "context": [
                "EXPORT COMPLIANCE. The software provided under this Agreement may be subject to export control laws and regulations.",
                "Customer shall comply with all applicable export and import laws and regulations in its use of the software.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
        {
            "category": "out_of_context",
            "domain": "medical",
            "question": "What diagnostic tests are available?",
            "answer": "A complete blood count measures red blood cells, white blood cells, and platelets. Comprehensive metabolic panels include glucose, electrolytes, and liver function tests. HbA1c testing provides a 3-month average blood glucose level.",
            "context": [
                "LAB RESULTS — Patient #4287: Sodium 140 mEq/L, Potassium 4.2 mEq/L, Chloride 102 mEq/L, Bicarbonate 24 mEq/L.",
                "BUN 15 mg/dL, Creatinine 1.0 mg/dL, Glucose 95 mg/dL. All values within normal reference ranges.",
            ],
            "sentence_labels": ["hallucinated", "hallucinated", "hallucinated"],
        },
    ]


def build_dataset() -> list[dict]:
    cases = []
    cases.extend(build_faithful_cases())
    cases.extend(build_number_substitution_cases())
    cases.extend(build_fabricated_claims_cases())
    cases.extend(build_partial_support_cases())
    cases.extend(build_subtle_contradiction_cases())
    cases.extend(build_out_of_context_cases())
    return cases


def evaluate_dataset(
    cases: list[dict],
    nli_model: str,
    trust_threshold: float,
) -> dict[str, Any]:
    category_metrics: dict[str, dict[str, Any]] = {}
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tn = 0
    all_predicted_scores: list[float] = []
    all_actual_labels: list[bool] = []
    all_latencies: list[float] = []

    for i, case in enumerate(cases):
        question = case["question"]
        answer = case["answer"]
        context = case["context"]
        gold_labels = case["sentence_labels"]

        start = time.time()
        result = verify(
            question=question,
            answer=answer,
            context=context,
            nli_model=nli_model,
            trust_threshold=trust_threshold,
        )
        latency = time.time() - start

        predicted_unsupported = {s.index for s in result.unsupported}

        for sent in result.sentences:
            all_predicted_scores.append(sent.trust_score)
            all_latencies.append(latency / max(len(result.sentences), 1))

        for j, gold in enumerate(gold_labels):
            if j >= len(result.sentences):
                continue
            is_hallucinated = gold == "hallucinated"
            predicted_flag = j in predicted_unsupported

            all_actual_labels.append(not is_hallucinated)

            cat = case["category"]
            if cat not in category_metrics:
                category_metrics[cat] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            cm = category_metrics[cat]

            if is_hallucinated and predicted_flag:
                cm["tp"] += 1
                all_tp += 1
            elif not is_hallucinated and predicted_flag:
                cm["fp"] += 1
                all_fp += 1
            elif is_hallucinated and not predicted_flag:
                cm["fn"] += 1
                all_fn += 1
            else:
                cm["tn"] += 1
                all_tn += 1

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(cases)} cases ({nli_model}, threshold={trust_threshold})")

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    per_category = {}
    for cat, cm in sorted(category_metrics.items()):
        p, r, f = _prf(cm["tp"], cm["fp"], cm["fn"])
        per_category[cat] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "confusion_matrix": dict(cm),
            "num_cases": sum(1 for c in cases if c["category"] == cat),
        }

    overall_p, overall_r, overall_f = _prf(all_tp, all_fp, all_fn)
    ece = compute_ece(all_predicted_scores, all_actual_labels)
    sorted_lat = sorted(all_latencies)

    return {
        "nli_model": nli_model,
        "trust_threshold": trust_threshold,
        "overall": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f1": round(overall_f, 4),
            "ece": round(ece, 4),
            "confusion_matrix": {
                "true_positives": all_tp,
                "false_positives": all_fp,
                "false_negatives": all_fn,
                "true_negatives": all_tn,
            },
            "num_cases": len(cases),
            "num_sentences": sum(len(c["sentence_labels"]) for c in cases),
            "latency_p50_ms": round(_percentile(sorted_lat, 0.50) * 1000, 1),
            "latency_p95_ms": round(_percentile(sorted_lat, 0.95) * 1000, 1),
        },
        "per_category": per_category,
    }


def print_markdown_table(results: list[dict[str, Any]]) -> None:
    print("\n## Overall Results\n")
    header = "| Model | Threshold | Precision | Recall | F1 | ECE | Latency p50 (ms) | Latency p95 (ms) |"
    sep = "|-------|-----------|-----------|--------|----|-----|-------------------|-------------------|"
    print(header)
    print(sep)
    for r in results:
        o = r["overall"]
        model_short = r["nli_model"].replace("cross-encoder/", "")
        print(
            f"| {model_short} | {r['trust_threshold']:.2f} "
            f"| {o['precision']:.4f} | {o['recall']:.4f} "
            f"| {o['f1']:.4f} | {o['ece']:.4f} "
            f"| {o['latency_p50_ms']:.1f} | {o['latency_p95_ms']:.1f} |"
        )

    best = max(results, key=lambda r: r["overall"]["f1"])
    best_model = best["nli_model"].replace("cross-encoder/", "")
    print(
        f"\n**Best configuration:** {best_model} at threshold {best['trust_threshold']:.2f} "
        f"(F1={best['overall']['f1']:.4f})"
    )

    print("\n## Per-Category Results (Best Configuration)\n")
    cats = sorted(best["per_category"].keys())
    cat_header = "| Category | Precision | Recall | F1 | Cases |"
    cat_sep = "|----------|-----------|--------|----|-------|"
    print(cat_header)
    print(cat_sep)
    for cat in cats:
        cm = best["per_category"][cat]
        print(
            f"| {cat} | {cm['precision']:.4f} "
            f"| {cm['recall']:.4f} | {cm['f1']:.4f} "
            f"| {cm['num_cases']} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Run full synthetic evaluation for athena-verify")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/full_eval.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80],
        help="Trust thresholds to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)

    print("Building synthetic benchmark dataset...")
    cases = build_dataset()
    category_counts = {}
    for c in cases:
        category_counts[c["category"]] = category_counts.get(c["category"], 0) + 1
    print(f"Total test cases: {len(cases)}")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    total_sentences = sum(len(c["sentence_labels"]) for c in cases)
    print(f"Total sentences: {total_sentences}")

    nli_models = [
        "cross-encoder/nli-deberta-v3-large",
        "cross-encoder/nli-deberta-v3-base",
    ]

    all_results: list[dict[str, Any]] = []
    for nli_model in nli_models:
        for threshold in args.thresholds:
            print(f"\nEvaluating: {nli_model}, threshold={threshold}")
            result = evaluate_dataset(cases, nli_model=nli_model, trust_threshold=threshold)
            all_results.append(result)

    output_data = {
        "dataset": {
            "total_cases": len(cases),
            "total_sentences": total_sentences,
            "categories": category_counts,
        },
        "configurations": all_results,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": args.seed,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print_markdown_table(all_results)


if __name__ == "__main__":
    main()
