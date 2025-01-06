entityextraction = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
Rule: Exclude any entity that represents a date, time, or temporal expression.
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_timestamp: timestamp of the relationship, if there is no obvious time information here, it can be left blank. format for timestamp:DD/MM/YYYY
- relationship_name: give the relationship a name
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
 
3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.
 
4. When finished, output {completion_delimiter}
 
######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
Step 1: Initial Consultation and Case Assessment

Date: October 4, 2023, 2:00 PMParticipants: Lawyer (John Smith), Plaintiff's Representative (Jane Doe, COO of ABC Manufacturing Corp.)

During the first meeting, the lawyer gathers all relevant details about the case. Jane Doe provides a detailed account of the disrupted deliveries and its financial impact. Key documents, including the signed contract, delivery schedules, and correspondence with XYZ Logistics, are handed over.
######################
Output:
("entity"{tuple_delimiter}ABC MANUFACTURING CORP.{tuple_delimiter}ORGANIZATION{tuple_delimiter}ABC Manufacturing Corp. is the plaintiff in a commercial lawsuit regarding a breach of contract case against XYZ Logistics LLC)
{record_delimiter}
("entity"{tuple_delimiter}JANE DOE{tuple_delimiter}PERSON{tuple_delimiter}Jane Doe is the COO of ABC Manufacturing Corp. and serves as the representative in the case)
{record_delimiter}
("entity"{tuple_delimiter}JOHN SMITH{tuple_delimiter}PERSON{tuple_delimiter}John Smith is the lawyer representing ABC Manufacturing Corp. in the case)
{record_delimiter}
("entity"{tuple_delimiter}XYZ LOGISTICS LLC{tuple_delimiter}ORGANIZATION{tuple_delimiter}XYZ Logistics LLC is the defendant in the commercial lawsuit accused of breaching a delivery contract)
{record_delimiter}
("relationship"{tuple_delimiter}ABC MANUFACTURING CORP.{tuple_delimiter}JANE DOE{tuple_delimiter}Jane Doe represents ABC Manufacturing Corp. in the commercial lawsuit{tuple_delimiter}14:00:00 10/04/2023{tuple_delimiter}represented by{tuple_delimiter}5)
{record_delimiter}
("relationship"{tuple_delimiter}ABC MANUFACTURING CORP.{tuple_delimiter}JOHN SMITH{tuple_delimiter}John Smith is the lawyer for ABC Manufacturing Corp. in the commercial lawsuit{tuple_delimiter}14:00:00 10/04/2023{tuple_delimiter}represented by{tuple_delimiter}5)
{record_delimiter}
("relationship"{tuple_delimiter}ABC MANUFACTURING CORP.{tuple_delimiter}XYZ LOGISTICS LLC{tuple_delimiter}XYZ Logistics LLC is being sued by ABC Manufacturing Corp. for breach of contract{tuple_delimiter}14:00:00 10/04/2023{tuple_delimiter}suing{tuple_delimiter}5)
{completion_delimiter}

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
{record_delimiter}
("entity"{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Vision Holdings is a firm that previously owned TechGlobal)
{record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}Vision Holdings formerly owned TechGlobal from 2014 until present{tuple_delimiter}00:00:00 01/01/2014{tuple_delimiter}owned by{tuple_delimiter}5)
{completion_delimiter}

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
September 6, 2024: Five Aurelians jailed for eight years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.
8:00 AM, September 6, 2024: The swap, orchestrated by Quintara, was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.
10:30 AM, September 6, 2024: The exchange was initiated in Firuzabad's capital, Tiruzia, leading to the four men and one woman—who are also Firuzi nationals—boarding a chartered flight to Krohaara.
1:15 PM, September 6, 2024: They were welcomed by senior Aurelian officials in Krohaara and immediately began their journey to Aurelia's capital, Cashion.
The group includes Samuel Namara, a 39-year-old businessman held in Tiruzia's Alhamia Prison, as well as Durke Bataglani, a 59-year-old journalist, and Meggie Tazbah, a 53-year-old environmentalist who also holds Bratinas nationality.
######################
Output:
("entity"{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Firuzabad held Aurelians as hostages)
{record_delimiter}
("entity"{tuple_delimiter}AURELIA{tuple_delimiter}GEO{tuple_delimiter}Country seeking to release hostages)
{record_delimiter}
("entity"{tuple_delimiter}QUINTARA{tuple_delimiter}GEO{tuple_delimiter}Country that negotiated a swap of money in exchange for hostages)
{record_delimiter}
{record_delimiter}
("entity"{tuple_delimiter}TIRUZIA{tuple_delimiter}GEO{tuple_delimiter}Capital of Firuzabad where the Aurelians were being held)
{record_delimiter}
("entity"{tuple_delimiter}KROHAARA{tuple_delimiter}GEO{tuple_delimiter}Capital city in Quintara)
{record_delimiter}
("entity"{tuple_delimiter}CASHION{tuple_delimiter}GEO{tuple_delimiter}Capital city in Aurelia)
{record_delimiter}
("entity"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}Aurelian who spent time in Tiruzia's Alhamia Prison)
{record_delimiter}
("entity"{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}GEO{tuple_delimiter}Prison in Tiruzia)
{record_delimiter}
("entity"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}Aurelian journalist who was held hostage)
{record_delimiter}
("entity"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}Bratinas national and environmentalist who was held hostage)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia{tuple_delimiter}06/09/2024{tuple_delimiter}HostageExchangeCoordinator{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}Samuel Namara was a prisoner at Alhamia prison{tuple_delimiter}06/09/2024{tuple_delimiter}ImprisonmentLocation{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}Samuel Namara and Meggie Tazbah were exchanged in the same hostage release{tuple_delimiter}06/09/2024{tuple_delimiter}HostageExchangeParticipants{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Samuel Namara and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}06/09/2024{tuple_delimiter}HostageExchangeParticipants{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}06/09/2024{tuple_delimiter}HostageExchangeParticipants{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Samuel Namara was a hostage in Firuzabad{tuple_delimiter}06/09/2024{tuple_delimiter}HostageLocation{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}FIRUZABAD{tuple_delimiter}Meggie Tazbah was a hostage in Firuzabad{tuple_delimiter}06/09/2024{tuple_delimiter}HostageLocation{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}FIRUZABAD{tuple_delimiter}Durke Bataglani was a hostage in Firuzabad{tuple_delimiter}06/09/2024{tuple_delimiter}HostageLocation{tuple_delimiter}2)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""
