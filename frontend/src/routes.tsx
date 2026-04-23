import { RootRoute, Route } from "@tanstack/react-router";
import { AppShell } from "./components";
import { CockpitPage, OpsPage, ResearchPage } from "./pages";

const rootRoute = new RootRoute({
  component: AppShell,
});

const cockpitRoute = new Route({
  getParentRoute: () => rootRoute,
  path: "/",
  component: CockpitPage,
});

const researchRoute = new Route({
  getParentRoute: () => rootRoute,
  path: "/research",
  component: ResearchPage,
});

const opsRoute = new Route({
  getParentRoute: () => rootRoute,
  path: "/ops",
  component: OpsPage,
});

export const routeTree = rootRoute.addChildren([cockpitRoute, researchRoute, opsRoute]);
