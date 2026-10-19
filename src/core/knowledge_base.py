"""
Local botanical knowledge base.

Loaded once at startup from data/knowledge/plant_knowledge.json.
Provides enriched knowledge cards for the 5 target plant classes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import KNOWLEDGE_DB_PATH

logger = logging.getLogger(__name__)

# ── Built-in knowledge for the 5 target plants ───────────────────────────────
_BUILTIN_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "colocasia_leaves": {
        "common_name": "Colocasia / Taro Leaves",
        "scientific_name": "Colocasia esculenta",
        "family": "Araceae",
        "description": (
            "Large, heart-shaped leaves with prominent central veins. "
            "Widely cultivated in tropical and subtropical Asia for its "
            "edible corms and leaves."
        ),
        "edibility": "Edible when cooked; raw leaves contain calcium oxalate crystals.",
        "cultivation": "Prefers wet, humid conditions; often grown near water bodies.",
        "nutritional_highlights": "Rich in dietary fibre, vitamin C, and potassium.",
        "fgvc_notes": (
            "Distinguished from similar Alocasia by peltate leaf attachment "
            "and matte underside surface. Leaf drip tip is more pronounced."
        ),
        "traditional_uses": [
            "Curries (Kerala sadya)",
            "Taro chips",
            "Leaf-wrapped preparations",
        ],
        "caution": "Must be cooked thoroughly to neutralise oxalates.",
    },
    "sorrel_leaves": {
        "common_name": "Green Sorrel / Sour Sorrel",
        "scientific_name": "Rumex acetosa",
        "family": "Polygonaceae",
        "description": (
            "Arrow-shaped leaves with a distinctly tart, lemony flavour "
            "owing to high oxalic acid content. Used extensively in "
            "French and Eastern European cuisines."
        ),
        "edibility": "Edible raw or cooked; best in moderation due to oxalic acid.",
        "cultivation": "Cool-season perennial; thrives in well-drained, slightly acidic soil.",
        "nutritional_highlights": "High in vitamin C, vitamin A, and antioxidants.",
        "fgvc_notes": (
            "Similar to spinach but leaves are narrower and arrow-shaped. "
            "Young leaves may resemble dock (Rumex obtusifolius)."
        ),
        "traditional_uses": [
            "Soupe à l'oseille (French sorrel soup)",
            "Sorrel chutney (Caribbean)",
            "Salads",
        ],
        "caution": "High oxalate content; avoid excessive consumption, especially with kidney conditions.",
    },
    "dill_leaves": {
        "common_name": "Dill Leaves / Dill Weed",
        "scientific_name": "Anethum graveolens",
        "family": "Apiaceae",
        "description": (
            "Feathery, thread-like blue-green fronds with a distinctive "
            "anise-like aroma. Both leaves and seeds are used as culinary herbs."
        ),
        "edibility": "Fully edible; widely used fresh or dried as a herb.",
        "cultivation": "Annual herb; grows best in full sun with well-drained soil.",
        "nutritional_highlights": "Contains flavonoids, monoterpenes (carvone, limonene), and calcium.",
        "fgvc_notes": (
            "Feathery pinnate structure distinguishes it from fennel fronds; "
            "fennel has a more pronounced licorice scent."
        ),
        "traditional_uses": [
            "Pickles (dill pickles)",
            "Dill raita",
            "Scandinavian gravlax",
            "Middle Eastern salads",
        ],
        "caution": "Generally safe; large medicinal doses may cause photosensitivity.",
    },
    "fenugreek_leaves": {
        "common_name": "Fenugreek Leaves / Methi",
        "scientific_name": "Trigonella foenum-graecum",
        "family": "Fabaceae",
        "description": (
            "Trifoliate leaves with a slightly bitter, maple-like aroma. "
            "One of the most important herbs in South Asian cuisine and Ayurvedic medicine."
        ),
        "edibility": "Edible raw or cooked; slight bitterness mellows on cooking.",
        "cultivation": "Cool-season annual; tolerates a range of soils; fast-growing.",
        "nutritional_highlights": "Excellent source of iron, calcium, and soluble fibre (galactomannan).",
        "fgvc_notes": (
            "Three rounded leaflets are distinctive; young leaves may be confused "
            "with clover, but fenugreek leaflets are obovate with finely toothed margins."
        ),
        "traditional_uses": [
            "Methi dal / methi paratha",
            "Sambar",
            "Kasuri methi (dried leaves)",
            "Ayurvedic blood-sugar management",
        ],
        "caution": "May interact with blood-thinning medication. Avoid in excess during pregnancy.",
    },
    "amaranth": {
        "common_name": "Green Amaranth / Chaulai",
        "scientific_name": "Amaranthus viridis",
        "family": "Amaranthaceae",
        "description": (
            "Nutrient-dense leafy green with soft, broad ovate leaves. "
            "Widely cultivated across Asia and Africa as a fast-growing staple potherb."
        ),
        "edibility": "Fully edible cooked or steamed; seeds are also edible as pseudo-grain.",
        "cultivation": "Warm-season annual; thrives in fertile, well-drained soil with full sunlight.",
        "nutritional_highlights": "Rich in calcium, iron, protein, dietary fibre, and vitamins A and C.",
        "fgvc_notes": (
            "Differentiated from spinach by lighter green hue and slight venation ribbing; "
            "differentiated from red amaranth by absence of betalain reddish-purple pigmentation."
        ),
        "traditional_uses": [
            "Chaulai saag",
            "Amaranth stir-fry with coconut (Thoran)",
            "Lentil and amaranth curries",
        ],
        "caution": "Contains mild levels of nitrates and oxalates; cooking reduces soluble oxalates.",
    },
    "red_amaranth": {
        "common_name": "Red Amaranth / Lal Chaulai",
        "scientific_name": "Amaranthus tricolor",
        "family": "Amaranthaceae",
        "description": (
            "Striking edible leafy green characterized by vibrant red, crimson, or purple "
            "leaf coloration due to high betalain antioxidant content."
        ),
        "edibility": "Fully edible raw (young microgreens) or cooked (mature foliage).",
        "cultivation": "Warm-season, drought-tolerant annual; develops intense red color under bright sun.",
        "nutritional_highlights": "Exceptional levels of betalain pigments, anthocyanins, iron, and vitamin K.",
        "fgvc_notes": (
            "Prominent central crimson-magenta pigmentation along veins distinguishes it "
            "readily from green amaranth and red spinach (Basella rubra).",
        ),
        "traditional_uses": [
            "Lal saag stir-fry",
            "Cheera thoran",
            "Nutritious vegetable broth",
        ],
        "caution": "Cook thoroughly to optimize mineral bioavailability and reduce oxalates.",
    },
    "coriander": {
        "common_name": "Coriander / Cilantro / Dhaniya",
        "scientific_name": "Coriandrum sativum",
        "family": "Apiaceae",
        "description": (
            "Delicate, aromatic herb with serrated, fan-shaped lower foliage and feathery upper leaves. "
            "Both fresh leaves (cilantro) and dried spherical schizocarp seeds (coriander spice) are used."
        ),
        "edibility": "Leaves, tender stems, roots, and seeds are all completely edible.",
        "cultivation": "Cool-season annual; sensitive to bolting in extreme heat; prefers moist, rich soil.",
        "nutritional_highlights": "High in linalool, vitamin K, vitamin C, and dietary antioxidants.",
        "fgvc_notes": (
            "Leaves have rounded serrations distinct from flat Italian parsley. "
            "Whole seeds are dry, spherical, and ribbed, presenting a unique fine-grained challenge vs foliage."
        ),
        "traditional_uses": [
            "Fresh garnish for curries and soups",
            "Coriander mint chutney",
            "Ground coriander seed in spice blends (Garam Masala)",
        ],
        "caution": "Certain individuals carry the OR6A2 olfactory receptor gene causing a soapy perception.",
    },
    "spinach": {
        "common_name": "Spinach",
        "scientific_name": "Spinacia oleracea",
        "family": "Amaranthaceae",
        "description": (
            "Smooth, dark green ovate to triangular leaves with a mild flavour. "
            "One of the world's most nutrient-dense leafy greens."
        ),
        "edibility": "Edible raw or cooked; widely consumed globally.",
        "cultivation": "Cool-season annual; prefers well-drained fertile soil.",
        "nutritional_highlights": "High in iron, folate, vitamins K, A, and C.",
        "fgvc_notes": (
            "Similar to Malabar spinach (Basella alba) which has thicker, "
            "succulent leaves and a woody stem. True spinach leaves are thin and flat."
        ),
        "traditional_uses": [
            "Palak paneer",
            "Saag",
            "Spinach salads",
            "Smoothies",
            "Pasta fillings",
        ],
        "caution": "High in oxalates; consume alongside calcium-rich foods to reduce absorption.",
    },
    "mint": {
        "common_name": "Mint / Pudina",
        "scientific_name": "Mentha spicata / Mentha piperita",
        "family": "Lamiaceae",
        "description": (
            "Fast-growing aromatic herb with bright-green serrated leaves and a distinctive "
            "cool menthol fragrance. Spearmint (Mentha spicata) is the most common Indian variety."
        ),
        "edibility": "Leaves and stems are fully edible; widely used fresh or dried.",
        "cultivation": "Thrives in moist, well-drained soil with partial shade. Spreads rapidly; best grown in containers.",
        "nutritional_highlights": "Rich in vitamins A and C, iron, manganese, rosmarinic acid, and antioxidants.",
        "fgvc_notes": (
            "Serrated leaf margins and opposite leaf arrangement are key visual identifiers. "
            "Leaves are smaller and more intensely green than coriander; stems are square in cross-section."
        ),
        "traditional_uses": [
            "Coriander-mint chutney",
            "Raita",
            "Mint tea / herbal infusions",
            "Mojitos and cold drinks",
            "Salad garnish",
            "Biriyani flavouring",
        ],
        "caution": "Generally safe; large medicinal doses may be unsuitable for infants and young children.",
    },
}


class KnowledgeBase:
    """Loads and queries the botanical knowledge database."""

    _db: dict[str, dict[str, Any]] = {}
    _loaded: bool = False

    def load(self, path: Path = KNOWLEDGE_DB_PATH) -> None:
        """Merge built-in knowledge with optional user-provided JSON."""
        self._db = dict(_BUILTIN_KNOWLEDGE)  # start with built-in

        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    user_data = json.load(f)
                
                # Support either dict mapping or list of card dicts
                if isinstance(user_data, list):
                    for item in user_data:
                        name = item.get("common_name", "") or item.get("scientific_name", "")
                        if name:
                            self._db[_normalise_key(name)] = item
                elif isinstance(user_data, dict):
                    for key, val in user_data.items():
                        normalised_key = _normalise_key(key)
                        if normalised_key in self._db:
                            self._db[normalised_key].update(val)
                        else:
                            self._db[normalised_key] = val
                logger.info("Loaded knowledge DB from %s", path)
            except Exception as exc:
                logger.warning("Could not load knowledge DB from %s: %s", path, exc)

        self._loaded = True
        logger.info("Knowledge base ready with %d plant entries.", len(self._db))

    def get(self, class_name: str) -> dict[str, Any]:
        """
        Retrieve a knowledge card for *class_name*.

        Falls back to a stub card if the class is not in the DB.
        """
        if not self._loaded:
            self.load()

        key = _normalise_key(class_name)

        # Exact match first
        if key in self._db:
            return self._db[key]

        # Fuzzy fallback: check if any DB key is a substring of the query
        for db_key, entry in self._db.items():
            if db_key in key or key in db_key:
                return entry

        return {
            "common_name": class_name.replace("_", " ").title(),
            "scientific_name": "Unknown",
            "description": "No detailed information available for this plant.",
            "note": "Consider adding an entry to data/knowledge/plant_knowledge.json",
        }

    def all_keys(self) -> list[str]:
        return list(self._db.keys())


def _normalise_key(name: str) -> str:
    """Lower-case, strip extra spaces, replace spaces/hyphens with underscores."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


# Module-level singleton
knowledge_base = KnowledgeBase()
