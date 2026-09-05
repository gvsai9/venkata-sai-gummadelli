import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const writing = defineCollection({
    loader: glob({
        pattern: "**/*.md",
        base: "./src/content/writing",
    }),

    schema: ({ image }) =>
        z
            .object({
                title: z.string(),
                description: z.string(),
                pubDate: z.coerce.date(),
                category: z.string(),
                tags: z.array(z.string()).optional(),

                // Optional thumbnail, used by article listing cards, the
                // homepage "Featured Writing" section, and future
                // related-article cards. Existing articles without a
                // thumbnail continue to work unchanged.
                thumbnail: image().optional(),
                thumbnailAlt: z.string().optional(),
            })
            .refine(
                (data) => !data.thumbnail || !!data.thumbnailAlt,
                {
                    message:
                        "thumbnailAlt is required whenever a thumbnail is set.",
                    path: ["thumbnailAlt"],
                }
            ),
});

const showcase = defineCollection({
    loader: glob({
        pattern: "**/*.md",
        base: "./src/content/showcase",
    }),

    schema: ({ image }) =>
        z
            .object({
                title: z.string(),

                // Drives the small kicker label on each card.
                type: z.enum([
                    "article",
                    "project",
                    "research",
                    "achievement",
                    "launch",
                    "update",
                ]),

                summary: z.string(),
                date: z.coerce.date().optional(),

                // Optional visual. When omitted, the showcase renders a
                // tasteful typographic placeholder instead of leaving the
                // card empty or inventing a stock image.
                image: image().optional(),
                imageAlt: z.string().optional(),

                // Where the whole card links to — an internal route
                // (e.g. "/writing/my-post" or "/work#my-project") or an
                // absolute external URL.
                href: z.string(),

                // Optional call-to-action label; falls back to a sensible
                // default per `type` if omitted (see FeaturedShowcase.astro).
                cta: z.string().optional(),

                // Optional manual sort order (ascending). Falls back to
                // `date` (newest first), then declaration order.
                order: z.number().optional(),
            })
            .refine((data) => !data.image || !!data.imageAlt, {
                message: "imageAlt is required whenever an image is set.",
                path: ["imageAlt"],
            }),
});

export const collections = {
    writing,
    showcase,
};
